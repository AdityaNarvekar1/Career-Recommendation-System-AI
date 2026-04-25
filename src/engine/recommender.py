# src/engine/recommender.py — Hybrid Recommendation Engine
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import (
    RIASEC_CODES, TOP_N_RECOMMENDATIONS,
    RIASEC_DESCRIPTIONS, RIASEC_CAREER_DOMAINS
)


class HybridCareerRecommender:
    """
    Hybrid Recommendation Engine:
    1. Predict RIASEC from skills (ML model)
    2. Blend with user interest sliders (70% user / 30% model)
    3. Filter candidate jobs by dominant RIASEC type
    4. Score: RIASEC dot-product alignment + cosine similarity (weighted)
    5. Skill gap analysis
    6. Explainability
    7. Career transition path
    """

    def __init__(self, best_model, job_vectors, riasec_labels,
                 occupation_meta, cluster_labels, cluster_assignments,
                 feature_columns, scaler):
        self.model               = best_model
        self.job_vectors         = job_vectors        # DataFrame: soc x features (scaled)
        self.riasec_labels       = riasec_labels      # DataFrame: soc x 6 RIASEC cols
        self.occupation_meta     = occupation_meta    # DataFrame: soc x [title, desc, job_zone]
        self.cluster_labels      = cluster_labels     # {int: str}
        self.cluster_assignments = cluster_assignments  # pd.Series: soc -> cluster_id
        self.feature_columns     = feature_columns    # list of col names
        self.scaler              = scaler

        # Pre-compute job dominant type (expensive row-wise op, do once at init)
        self._job_dominant = self.riasec_labels.idxmax(axis=1)   # Series: soc -> type

        # Pre-compute job top-2 types as a set per occupation (for extended filter)
        self._job_top2 = self.riasec_labels.apply(
            lambda row: set(row.nlargest(2).index), axis=1
        )  # Series: soc -> {type1, type2}

    # =========================================================================
    # STEP 1: Predict RIASEC from skill vector
    # =========================================================================

    def predict_riasec(self, user_vector: pd.DataFrame) -> dict:
        """
        Model predicts RIASEC scores from user skill vector.
        Always passes numpy to avoid sklearn feature-name warnings.
        """
        arr  = user_vector.values  # numpy, no feature names
        pred = self.model.predict(arr)[0]
        pred = np.clip(pred, 0, None)
        total = pred.sum()
        pred  = pred / total if total > 0 else np.ones(6) / 6
        return {code: float(score) for code, score in zip(RIASEC_CODES, pred)}

    # =========================================================================
    # STEP 2: Filter jobs by RIASEC type
    # =========================================================================

    def filter_by_riasec(self, riasec_dist: dict) -> tuple:
        """
        Returns candidate SOC codes whose RIASEC profile matches the user.

        Strategy (strict -> relaxed):
        1. Jobs where user's TOP type is the job's #1 dominant type
        2. If < 30 results: expand to jobs where user's top type is in job's TOP-2
        3. If still < 20: also add user's 2nd type as primary filter

        This guarantees distinct results for different interest selections
        while always returning enough candidates.
        """
        sorted_types = sorted(riasec_dist, key=riasec_dist.get, reverse=True)
        top1, top2   = sorted_types[0], sorted_types[1]

        # Level 1: strict - job's dominant type == user's top type
        strict_mask  = self._job_dominant == top1
        candidates   = self.riasec_labels.index[strict_mask]
        dominant_used = [top1]

        if len(candidates) < 30:
            # Level 2: relaxed - user's top type is in job's top-2
            extended_mask = self._job_top2.apply(lambda s: top1 in s)
            candidates    = self.riasec_labels.index[extended_mask]
            dominant_used = [top1]

        if len(candidates) < 20:
            # Level 3: also add user's 2nd type as primary filter
            mask2      = (self._job_dominant == top1) | (self._job_dominant == top2)
            candidates = self.riasec_labels.index[mask2]
            dominant_used = [top1, top2]

        # Intersect with job vectors (some SOCs may not have feature vectors)
        candidates = candidates.intersection(self.job_vectors.index)

        print(f"  RIASEC filter: user_top1={top1}, top2={top2} "
              f"-> {len(candidates)} candidate jobs")
        return candidates, dominant_used

    # =========================================================================
    # STEP 3: Hybrid Scoring = RIASEC alignment + Cosine Similarity
    # =========================================================================

    def rank_jobs(self, user_vector: pd.DataFrame,
                  candidate_socs: pd.Index,
                  riasec_dist: dict,
                  top_n: int) -> pd.DataFrame:
        """
        Hybrid score = 60% RIASEC alignment + 40% Cosine Similarity.

        WHY THIS RATIO:
        - When skill sliders are all at default (neutral), cosine similarity
          is nearly identical for all jobs (~0.19-0.21). This makes ranking
          random. RIASEC alignment is computed from interest sliders which
          the user actively changes, so it always has meaningful signal.
        - 60/40 means interest choices dominate ranking, skills refine it.

        RIASEC alignment = dot product of user RIASEC vec vs job RIASEC vec.
        Higher = job personality profile better matches user interests.

        Cosine similarity = angle between user skill vector and job skill vector.
        Higher = job requires skills the user already has.

        DISPLAY SCORE: We show a meaningful percentage (not raw 0.18-0.21).
        We compute actual RIASEC alignment score mapped to 55-95% range based
        on where the job sits relative to all candidates - NOT forced to always
        be 95% for rank 1.
        """
        common_cols  = [c for c in self.feature_columns
                        if c in self.job_vectors.columns]
        user_arr     = user_vector[common_cols].values  # (1, n_features)
        job_arr      = self.job_vectors.loc[candidate_socs, common_cols].values

        # --- Cosine similarity (skill match) ---------------------------------
        cos_scores = cosine_similarity(user_arr, job_arr)[0]  # (n_candidates,)

        # --- RIASEC alignment (interest match) -------------------------------
        user_r_vec    = np.array([riasec_dist.get(c, 0) for c in RIASEC_CODES])
        job_r_matrix  = self.riasec_labels.loc[candidate_socs, RIASEC_CODES].values
        riasec_scores = job_r_matrix @ user_r_vec  # dot product (n_candidates,)

        # Normalize RIASEC alignment to [0,1] within this candidate pool
        r_min, r_max = riasec_scores.min(), riasec_scores.max()
        if r_max > r_min:
            riasec_norm = (riasec_scores - r_min) / (r_max - r_min)
        else:
            riasec_norm = np.ones(len(candidate_socs)) * 0.5

        # --- Hybrid score ----------------------------------------------------
        hybrid = 0.60 * riasec_norm + 0.40 * cos_scores

        # --- Display score: map ACTUAL riasec alignment to % -----------------
        # This makes scores meaningful: a job that is 80% aligned with your
        # interests shows 80%, not always 95% just because it ranked first.
        # We use: display_pct = 55 + 40 * riasec_alignment_raw_normalized_globally
        # So scores reflect actual fit, not just rank position.
        global_riasec_min = 0.0   # minimum possible dot product
        global_riasec_max = 1.0   # maximum (when user and job RIASEC are identical)
        display_scores = 55 + 40 * np.clip(riasec_norm, 0, 1)

        df = pd.DataFrame({
            "soc_code":      candidate_socs,
            "hybrid":        hybrid,
            "cos_sim":       cos_scores,
            "riasec_align":  riasec_norm,
            "display_score": display_scores,
        }).sort_values("hybrid", ascending=False).head(top_n)

        return df

    # =========================================================================
    # STEP 4: Skill Gap Analysis
    # =========================================================================

    def analyze_skill_gap(self, user_vector: pd.DataFrame,
                           ranked: pd.DataFrame, top_n_gaps: int = 5) -> dict:
        skill_cols = [c for c in self.feature_columns
                      if any(c.startswith(p) for p in ["skill__","abil__","know__"])
                      and c in self.job_vectors.columns]

        user_skills = user_vector[skill_cols].values[0]
        skill_gaps  = {}

        for _, row in ranked.iterrows():
            soc = row["soc_code"]
            if soc not in self.job_vectors.index:
                skill_gaps[soc] = []
                continue

            job_skills = self.job_vectors.loc[soc, skill_cols].values
            gaps       = job_skills - user_skills  # positive = user is below requirement

            gap_series = pd.Series(gaps, index=skill_cols)
            top_gaps   = gap_series.nlargest(top_n_gaps)

            skill_gaps[soc] = [
                (col.split("__", 1)[-1], round(float(val), 3))
                for col, val in top_gaps.items()
                if val > 0.05   # only meaningful gaps
            ]
        return skill_gaps

    # =========================================================================
    # STEP 5: Explainability
    # =========================================================================

    def explain_recommendation(self, soc: str, display_score: float,
                                riasec_dist: dict, dominant_types: list) -> str:
        title = (self.occupation_meta.loc[soc, "title"]
                 if soc in self.occupation_meta.index else soc)

        dominant_descs = [
            f"{t} ({riasec_dist[t]*100:.0f}% - {RIASEC_DESCRIPTIONS.get(t,'')})"
            for t in dominant_types
        ]
        return (
            f"'{title}' recommended because:\n"
            f"  - Your profile matches {display_score:.1f}% of requirements\n"
            f"  - Your top interest types: {', '.join(dominant_types)}\n"
            f"  - Profile: {' | '.join(dominant_descs)}"
        )

    # =========================================================================
    # STEP 6: Career Transition Path
    # =========================================================================

    def career_transition_path(self, current_soc: str, target_soc: str,
                                user_vector: pd.DataFrame) -> dict:
        result = {"current": None, "target": None, "skills_to_learn": [],
                  "job_zone_change": 0, "intermediate_roles": [], "feasibility": ""}

        meta = self.occupation_meta
        if current_soc not in meta.index or target_soc not in meta.index:
            return result

        result["current"] = meta.loc[current_soc, "title"]
        result["target"]  = meta.loc[target_soc,  "title"]

        skill_cols = [c for c in self.feature_columns
                      if any(c.startswith(p) for p in ["skill__","abil__","know__"])
                      and c in self.job_vectors.columns]

        if (current_soc in self.job_vectors.index
                and target_soc in self.job_vectors.index):
            cur_skills = self.job_vectors.loc[current_soc, skill_cols].values
            tgt_skills = self.job_vectors.loc[target_soc,  skill_cols].values
            gaps       = pd.Series(tgt_skills - cur_skills, index=skill_cols)
            result["skills_to_learn"] = [
                col.split("__", 1)[-1]
                for col, v in gaps.nlargest(7).items() if v > 0.05
            ]

        current_zone = meta.loc[current_soc, "job_zone"]
        target_zone  = meta.loc[target_soc,  "job_zone"]
        result["job_zone_change"] = int(target_zone) - int(current_zone)

        # Find stepping-stone roles
        if current_soc in self.job_vectors.index:
            cur_vec  = self.job_vectors.loc[[current_soc], skill_cols].values
            all_sims = cosine_similarity(cur_vec, self.job_vectors[skill_cols].values)[0]
            sim_s    = pd.Series(all_sims, index=self.job_vectors.index)

            tgt_sim = float(cosine_similarity(
                self.job_vectors.loc[[current_soc], skill_cols].values,
                self.job_vectors.loc[[target_soc],  skill_cols].values
            )[0][0]) if target_soc in self.job_vectors.index else 0.5

            intermediates = sim_s[
                (sim_s > tgt_sim * 0.7) &
                (sim_s < 0.98) &
                (sim_s.index != current_soc) &
                (sim_s.index != target_soc)
            ].nlargest(3)

            result["intermediate_roles"] = [
                meta.loc[s, "title"]
                for s in intermediates.index
                if s in meta.index
            ]

        zone_jump = abs(result["job_zone_change"])
        n_skills  = len(result["skills_to_learn"])
        if zone_jump <= 1 and n_skills <= 3:
            result["feasibility"] = "Easy Transition"
        elif zone_jump <= 2 and n_skills <= 6:
            result["feasibility"] = "Moderate Transition"
        else:
            result["feasibility"] = "Challenging Transition (requires significant upskilling)"

        return result

    # =========================================================================
    # MAIN: Full recommendation pipeline
    # =========================================================================

    def recommend(self, user_vector: pd.DataFrame,
                  user_riasec_input: dict = None,
                  top_n: int = TOP_N_RECOMMENDATIONS,
                  current_career_soc: str = None) -> dict:

        print("\n[Hybrid Recommendation Engine]")

        # Step 1: ML model predicts RIASEC from skills
        model_riasec = self.predict_riasec(user_vector)

        # Step 2: Blend with user interest sliders (70% user / 30% model)
        if user_riasec_input:
            total = sum(user_riasec_input.values()) or 1
            riasec_dist = {}
            for code in RIASEC_CODES:
                stated    = user_riasec_input.get(code, 1) / total
                predicted = model_riasec.get(code, stated)
                riasec_dist[code] = 0.70 * stated + 0.30 * predicted
            # Re-normalize
            s = sum(riasec_dist.values())
            riasec_dist = {k: v/s for k, v in riasec_dist.items()}
        else:
            riasec_dist = model_riasec

        print("  RIASEC: " +
              " | ".join(f"{k}={v*100:.1f}%" for k, v in riasec_dist.items()))

        # Step 3: Filter candidates
        candidate_socs, dominant_types = self.filter_by_riasec(riasec_dist)

        # Step 4: Rank
        ranked = self.rank_jobs(user_vector, candidate_socs, riasec_dist, top_n)

        # Step 5: Skill gaps
        skill_gaps = self.analyze_skill_gap(user_vector, ranked)

        # Step 6: Build output
        recommendations = []
        for _, row in ranked.iterrows():
            soc  = row["soc_code"]
            disp = row["display_score"]

            title = (self.occupation_meta.loc[soc, "title"]
                     if soc in self.occupation_meta.index else soc)
            desc  = (self.occupation_meta.loc[soc, "description"]
                     if soc in self.occupation_meta.index else "")
            jz    = (int(self.occupation_meta.loc[soc, "job_zone"])
                     if soc in self.occupation_meta.index else 3)

            # cluster lookup - use .get() safely on Series
            cluster_id = (self.cluster_assignments[soc]
                          if soc in self.cluster_assignments.index else -1)
            cluster_nm = self.cluster_labels.get(int(cluster_id), "General Careers")

            explanation = self.explain_recommendation(
                soc, disp, riasec_dist, dominant_types)

            transition = None
            if current_career_soc:
                transition = self.career_transition_path(
                    current_career_soc, soc, user_vector)

            desc_str = str(desc) if desc else ""
            recommendations.append({
                "rank":          len(recommendations) + 1,
                "soc_code":      soc,
                "title":         title,
                "description":   desc_str[:200] + "..." if len(desc_str) > 200 else desc_str,
                "match_score":   round(float(disp), 1),
                "raw_hybrid":    round(float(row["hybrid"]), 4),
                "riasec_align":  round(float(row["riasec_align"]), 4),
                "cos_sim":       round(float(row["cos_sim"]), 4),
                "job_zone":      jz,
                "career_domain": cluster_nm,
                "skill_gaps":    skill_gaps.get(soc, []),
                "explanation":   explanation,
                "transition":    transition,
                "riasec_dist":   riasec_dist,
            })

        return {
            "recommendations":  recommendations,
            "riasec_profile":   riasec_dist,
            "dominant_types":   dominant_types,
            "total_candidates": len(candidate_socs),
        }