import pandas as pd
import numpy as np
import warnings
from collections import Counter

warnings.filterwarnings("ignore")


#  CONFIGURATION

JOBROLE_GROUPS = {
    'Nurse': 'Clinical',
    'Therapist': 'Clinical',
    'Administrative': 'Administrative',
    'Admin': 'Administrative',
    'Other': 'Other'
}

NUMERIC_FEATURES = [
    "JobSatisfaction", "WorkLifeBalance", "MonthlyIncome", "JobInvolvement",
    "TrainingTimesLastYear", "DistanceFromHome", "EnvironmentSatisfaction",
    "RelationshipSatisfaction", "PerformanceRating", "YearsSinceLastPromotion",
    "YearsInCurrentRole", "TotalWorkingYears", "JobLevel", "OverTime",
    "Age", "NumCompaniesWorked", "Education", "YearsAtCompany"
]

CATEGORICAL_FEATURES = ["MaritalStatus", "Department", "BusinessTravel", "EducationField"]

FEATURES_FOR_STRATEGY = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# All four risk labels
HIGH_RISK_LABELS = ['High Risk', 'Critical Risk']
LOW_RISK_LABELS = ['Low Risk', 'Medium Risk']


#  STRATEGY MAPS

ROLE_BASED_STRATEGY_MAP = {

    "Clinical": {
        "JobSatisfaction": {"Low": ["Immediate clinical workload review", "Schedule one-on-one manager check-in",
                                    "Review task allocation and role fit"],
                            "Medium": ["Peer support discussions", "Monthly satisfaction pulse survey"],
                            "High": ["Clinical excellence recognition", "Nominate for internal awards program"]},
        "WorkLifeBalance": {"Low": ["Redesign hospital shift rotations", "Introduce mandatory rest periods",
                                    "Reduce consecutive night shifts"],
                            "Medium": ["Rotational off-days", "Review shift scheduling fairness"],
                            "High": ["Maintain structured duty cycles", "Share best practices across wards"]},
        "MonthlyIncome": {"Low": ["Clinical pay adjustment review", "Benchmark against market rates",
                                  "Apply for clinical pay band upgrade"],
                          "Medium": ["Incentive allowances", "Performance-linked pay review"],
                          "High": ["Performance-based bonuses", "Retention bonus for senior clinicians"]},
        "JobInvolvement": {"Low": ["Assign supervised case handling", "Introduce structured role goals",
                                   "Buddy system with senior clinician"],
                           "Medium": ["Team patient rounds", "Involve in clinical committees"],
                           "High": ["Lead clinical reviews", "Mentor junior clinical staff"]},
        "TrainingTimesLastYear": {"Low": ["Mandatory clinical training", "Enrol in skills refresher program",
                                          "Schedule quarterly CPD sessions"],
                                  "Medium": ["Medical education programs", "External conference sponsorship"],
                                  "High": ["Specialization sponsorship", "Research fellowship opportunities"]},
        "DistanceFromHome": {"Low": ["Hospital accommodation support", "Subsidised staff housing review"],
                             "Medium": ["Shift clustering to reduce commute", "Transport allowance review"],
                             "High": ["Relocation allowance", "Remote consultation options where possible"]},
        "EnvironmentSatisfaction": {
            "Low": ["Improve ward conditions", "Immediate facilities complaint review", "Staff wellbeing audit"],
            "Medium": ["Equipment upgrade review", "Introduce ward improvement suggestions box"],
            "High": ["Maintain high-quality facilities", "Celebrate ward achievements"]},
        "RelationshipSatisfaction": {
            "Low": ["Conflict mediation with HR", "Introduce anonymous feedback channel", "Team cohesion workshop"],
            "Medium": ["Team-building rounds", "Cross-ward social events"],
            "High": ["Interdisciplinary collaboration projects", "Peer recognition program"]},
        "PerformanceRating": {"Low": ["Clinical mentoring program", "Structured performance improvement plan",
                                      "Assign performance coach"],
                              "Medium": ["Case review discussions", "Quarterly performance check-ins"],
                              "High": ["Clinical excellence awards", "Fast-track to senior clinical role"]},
        "YearsSinceLastPromotion": {
            "Low": ["Promotion board review", "Clarify promotion criteria", "Career ladder mapping session"],
            "Medium": ["Structured growth pathway", "Identify next promotion milestone"],
            "High": ["Fast-track consultant pathway", "Senior role shadowing opportunity"]},
        "YearsInCurrentRole": {"Low": ["Department rotation program", "Cross-training in adjacent specialties"],
                               "Medium": ["Skill diversification plan", "Lateral move discussion"],
                               "High": ["Senior role enrichment", "Clinical leadership responsibilities"]},
        "TotalWorkingYears": {"Low": ["Career planning session", "Graduate development pathway"],
                              "Medium": ["Retention allowance review", "Mid-career development plan"],
                              "High": ["Senior retention incentives", "Long-service award and recognition"]},
        "JobLevel": {"Low": ["Role evaluation review", "Job level upgrade discussion", "Expand responsibilities"],
                     "Medium": ["Responsibility expansion plan", "Prepare for next level role"],
                     "High": ["Clinical leadership training", "Succession planning inclusion"]},
        "OverTime": {"Low": ["Reduce night shift frequency", "Review shift distribution fairness"],
                     "Medium": ["Balanced patient allocation", "Monitor overtime hours monthly"],
                     "High": ["Burnout monitoring program", "Mandatory rest period enforcement",
                              "Occupational health referral"]},
        "Age": {"Low": ["Graduate mentorship program", "Early career development plan", "Assign senior mentor"],
                "Medium": ["Mid-career development plan", "Leadership readiness assessment"],
                "High": ["Senior staff recognition", "Knowledge transfer program"]},
        "NumCompaniesWorked": {"Low": ["Loyalty rewards program", "Long-service recognition plan"],
                               "Medium": ["Career stability planning", "Retention milestone rewards"],
                               "High": ["Job-hopper engagement plan", "Career anchoring discussion"]},
        "Education": {"Low": ["Sponsored further education", "Tuition support program"],
                      "Medium": ["Advanced certification support", "Postgraduate study allowance"],
                      "High": ["Research opportunity access", "Fellowship and grant applications"]},
        "YearsAtCompany": {"Low": ["Onboarding buddy system", "Early integration support plan"],
                           "Medium": ["Tenure milestone rewards", "Mid-tenure career review"],
                           "High": ["Long-service recognition", "Senior retention package"]},
    },

    "Administrative": {
        "JobSatisfaction": {
            "Low": ["Task redistribution", "Role clarity workshop", "One-on-one satisfaction discussion"],
            "Medium": ["Feedback sessions", "Team morale check-in"],
            "High": ["Recognition programs", "Peer appreciation initiative"]},
        "WorkLifeBalance": {
            "Low": ["Office workload balancing", "Introduce task prioritisation framework", "Review meeting load"],
            "Medium": ["Flexible scheduling options", "Introduce no-meeting afternoons"],
            "High": ["Remote work support", "Flexible hours policy"]},
        "MonthlyIncome": {"Low": ["Compensation review", "Pay equity audit", "Benchmark against admin market rates"],
                          "Medium": ["Incentive alignment review", "Performance-linked pay plan"],
                          "High": ["Performance bonus", "Annual pay progression review"]},
        "JobInvolvement": {
            "Low": ["Responsibility expansion", "Assign cross-team project", "Introduce ownership over key process"],
            "Medium": ["Cross-department collaboration", "Involve in strategic planning"],
            "High": ["Leadership development program", "Include in senior management meetings"]},
        "TrainingTimesLastYear": {"Low": ["Administrative workshops", "Enrol in professional skills program",
                                          "Monthly lunch-and-learn sessions"],
                                  "Medium": ["Management training", "External leadership course sponsorship"],
                                  "High": ["Executive training program", "MBA or postgrad sponsorship"]},
        "DistanceFromHome": {"Low": ["Flexible start and end times", "Review commute support options"],
                             "Medium": ["Hybrid work policy", "Transport allowance review"],
                             "High": ["Relocation assistance package", "Remote work arrangement"]},
        "EnvironmentSatisfaction": {
            "Low": ["Office improvement initiative", "Staff environment survey", "Quick wins workspace upgrade"],
            "Medium": ["Workspace redesign project", "Introduce ergonomic equipment"],
            "High": ["Maintain positive workplace culture", "Culture ambassador program"]},
        "RelationshipSatisfaction": {
            "Low": ["HR mediation session", "Anonymous feedback mechanism", "Team conflict resolution workshop"],
            "Medium": ["Team engagement activities", "Cross-department social events"],
            "High": ["Leadership networking events", "Peer mentoring program"]},
        "PerformanceRating": {
            "Low": ["Performance coaching sessions", "Structured improvement plan", "Weekly progress check-ins"],
            "Medium": ["Improvement plan with milestones", "Quarterly review meetings"],
            "High": ["High-performance incentives", "Fast-track to senior admin role"]},
        "YearsSinceLastPromotion": {"Low": ["Promotion review meeting", "Document promotion eligibility criteria"],
                                    "Medium": ["Career planning session", "Identify promotion timeline"],
                                    "High": ["Fast-track management program", "Senior role shadowing"]},
        "YearsInCurrentRole": {"Low": ["Role rotation opportunity", "Cross-functional project assignment"],
                               "Medium": ["Responsibility enrichment plan", "Lateral career move discussion"],
                               "High": ["Senior management pathway", "Department head readiness program"]},
        "TotalWorkingYears": {"Low": ["Retention planning session", "Early career support program"],
                              "Medium": ["Retention incentives review", "Mid-career progression plan"],
                              "High": ["Executive retention package", "Long-service recognition"]},
        "JobLevel": {"Low": ["Level reassessment discussion", "Define next level criteria"],
                     "Medium": ["Responsibility expansion plan", "Prepare for level progression"],
                     "High": ["Succession planning inclusion", "Leadership readiness program"]},
        "OverTime": {"Low": ["Reduce admin overload", "Delegate lower-priority tasks"],
                     "Medium": ["Shift redistribution plan", "Introduce workload tracking"],
                     "High": ["Monitor workload sustainability", "Introduce delegation framework",
                              "Occupational health referral"]},
        "Age": {"Low": ["Graduate admin program", "Early career mentoring"],
                "Medium": ["Mid-career leadership path", "Management readiness program"],
                "High": ["Senior advisor recognition", "Knowledge transfer initiative"]},
        "NumCompaniesWorked": {"Low": ["Loyalty recognition program", "Long-service milestone celebration"],
                               "Medium": ["Stability incentives", "Career anchoring discussion"],
                               "High": ["Re-engagement program", "Job-hopper retention plan"]},
        "Education": {"Low": ["Admin skills training", "Professional development funding"],
                      "Medium": ["Management degree support", "Leadership certification sponsorship"],
                      "High": ["Executive education programs", "MBA sponsorship"]},
        "YearsAtCompany": {"Low": ["Induction support program", "New joiner buddy system"],
                           "Medium": ["Milestone recognition rewards", "Mid-tenure career review"],
                           "High": ["Long-service benefits", "Senior retention package"]},
    },

    "Other": {
        "JobSatisfaction": {
            "Low": ["One-on-one satisfaction review", "Role alignment discussion", "Reassess job responsibilities"],
            "Medium": ["Regular manager check-ins", "Role enrichment conversation"],
            "High": ["Peer recognition program", "Document and share success stories"]},
        "WorkLifeBalance": {"Low": ["Workload audit", "Introduce task prioritisation", "Reduce non-essential duties"],
                            "Medium": ["Flexible hours policy", "Review workload distribution"],
                            "High": ["Sustain current balance", "Share best practices with team"]},
        "MonthlyIncome": {"Low": ["Pay equity review", "Market benchmarking exercise"],
                          "Medium": ["Performance-linked pay plan", "Incentive structure review"],
                          "High": ["Retention bonus", "Annual pay progression"]},
        "JobInvolvement": {
            "Low": ["Assign meaningful projects", "Introduce role ownership goals", "Buddy with engaged colleague"],
            "Medium": ["Cross-team involvement", "Collaborative project assignment"],
            "High": ["Innovation contributor role", "Lead special initiative"]},
        "TrainingTimesLastYear": {
            "Low": ["General skills training", "Enrol in role-specific course", "Quarterly learning plan"],
            "Medium": ["Role-specific development program", "External training sponsorship"],
            "High": ["Advanced learning sponsorship", "Conference attendance support"]},
        "DistanceFromHome": {"Low": ["Commute support allowance", "Flexible start time options"],
                             "Medium": ["Hybrid work option", "Transport subsidy review"],
                             "High": ["Relocation package", "Remote work arrangement"]},
        "EnvironmentSatisfaction": {"Low": ["Workspace improvement plan", "Staff environment survey"],
                                    "Medium": ["Environment feedback sessions", "Quick wins improvement plan"],
                                    "High": ["Maintain positive environment", "Celebrate team workspace wins"]},
        "RelationshipSatisfaction": {
            "Low": ["Conflict resolution support", "HR mediation if needed", "Anonymous feedback channel"],
            "Medium": ["Team bonding activities", "Cross-team social events"],
            "High": ["Peer mentoring program", "Cross-functional networking"]},
        "PerformanceRating": {"Low": ["Performance support plan", "Assign performance coach", "Weekly progress review"],
                              "Medium": ["Goal-setting sessions", "Quarterly performance check-ins"],
                              "High": ["High-achiever recognition", "Fast-track career opportunity"]},
        "YearsSinceLastPromotion": {"Low": ["Promotion eligibility review", "Clarify promotion criteria"],
                                    "Medium": ["Career roadmap planning", "Identify next milestone"],
                                    "High": ["Fast-track consideration", "Senior role opportunity"]},
        "YearsInCurrentRole": {"Low": ["Role enrichment plan", "Cross-training opportunity"],
                               "Medium": ["Lateral move opportunities", "Skill diversification plan"],
                               "High": ["Senior contributor pathway", "Leadership readiness program"]},
        "TotalWorkingYears": {"Low": ["Early career mentoring", "Graduate development pathway"],
                              "Medium": ["Mid-career retention plan", "Career progression review"],
                              "High": ["Senior retention package", "Long-service recognition"]},
        "JobLevel": {"Low": ["Level advancement review", "Define next level expectations"],
                     "Medium": ["Expanded responsibilities", "Prepare for level progression"],
                     "High": ["Leadership readiness program", "Succession planning inclusion"]},
        "OverTime": {"Low": ["Overtime reduction plan", "Review task delegation"],
                     "Medium": ["Workload rebalancing", "Flexible scheduling options"],
                     "High": ["Burnout prevention check", "Wellness program referral", "Occupational health review"]},
        "Age": {"Low": ["Early career support program", "Assign senior mentor"],
                "Medium": ["Career growth planning", "Mid-career development review"],
                "High": ["Experience retention plan", "Knowledge transfer program"]},
        "NumCompaniesWorked": {"Low": ["Loyalty incentive program", "Long-service milestone recognition"],
                               "Medium": ["Career anchoring support", "Stability incentives"],
                               "High": ["Engagement and belonging plan", "Job-hopper retention strategy"]},
        "Education": {"Low": ["Upskilling opportunities", "Professional development funding"],
                      "Medium": ["Certification support", "Advanced training sponsorship"],
                      "High": ["Advanced study sponsorship", "Research or fellowship access"]},
        "YearsAtCompany": {"Low": ["New hire integration plan", "Onboarding buddy system"],
                           "Medium": ["Tenure-based rewards", "Mid-tenure career review"],
                           "High": ["Long-service appreciation", "Senior retention package"]},
    }
}


LOW_RISK_STRATEGY_MAP = {
    "JobSatisfaction": {"Low": ["Conduct stay interviews to identify early dissatisfiers"],
                        "Medium": ["Quarterly engagement check-ins"],
                        "High": ["Peer recognition and appreciation program"]},
    "WorkLifeBalance": {"Low": ["Flexible work schedule review and adjustment"],
                        "Medium": ["Optional wellness and mindfulness programs"],
                        "High": ["Sustain and reinforce current balance practices"]},
    "MonthlyIncome": {"Low": ["Proactive compensation benchmarking review"],
                      "Medium": ["Performance-linked pay discussion"],
                      "High": ["Retention bonus planning for continued loyalty"]},
    "JobInvolvement": {"Low": ["Assign stretch projects to increase engagement"],
                       "Medium": ["Cross-team collaboration opportunities"],
                       "High": ["Innovation contributor or internal champion role"]},
    "TrainingTimesLastYear": {"Low": ["Enroll in foundational skill development programs"],
                              "Medium": ["Role-specific certification support"],
                              "High": ["Advanced learning and conference sponsorship"]},
    "DistanceFromHome": {"Low": ["Commute subsidy or hybrid working option"],
                         "Medium": ["Flexible start and end time policy"],
                         "High": ["Maintain and reinforce current arrangement"]},
    "EnvironmentSatisfaction": {"Low": ["Workspace feedback survey and follow-up"],
                                "Medium": ["Environment improvement initiatives"],
                                "High": ["Maintain positive workplace environment"]},
    "RelationshipSatisfaction": {"Low": ["Team bonding and social activities"], "Medium": ["Peer mentoring program"],
                                 "High": ["Leadership networking and visibility events"]},
    "PerformanceRating": {"Low": ["Proactive goal-setting and coaching session"],
                          "Medium": ["Bi-annual performance review and feedback"],
                          "High": ["High-achiever spotlight and recognition"]},
    "YearsSinceLastPromotion": {"Low": ["Promotion eligibility and timeline discussion"],
                                "Medium": ["Career roadmap and growth planning"],
                                "High": ["Fast-track consideration for leadership roles"]},
    "YearsInCurrentRole": {"Low": ["Role enrichment and challenge discussion"],
                           "Medium": ["Lateral move or cross-functional opportunities"],
                           "High": ["Senior contributor or subject-matter expert pathway"]},
    "TotalWorkingYears": {"Low": ["Early career mentoring and guidance program"],
                          "Medium": ["Mid-career engagement and development plan"],
                          "High": ["Senior retention and knowledge-transfer package"]},
    "JobLevel": {"Low": ["Level advancement roadmap and timeline"],
                 "Medium": ["Expanded responsibilities and visibility"],
                 "High": ["Leadership readiness and succession planning"]},
    "OverTime": {"Low": ["Proactive workload monitoring to prevent overload"],
                 "Medium": ["Workload rebalancing check-in"], "High": ["Burnout prevention and early warning check"]},
    "Age": {"Low": ["Early career development and graduate program"],
            "Medium": ["Career growth and mid-career planning"],
            "High": ["Experience retention and knowledge-sharing plan"]},
    "NumCompaniesWorked": {"Low": ["Loyalty incentive and recognition program"],
                           "Medium": ["Career anchoring and stability support"],
                           "High": ["Belonging, engagement, and long-term plan"]},
    "Education": {"Low": ["Upskilling and foundational learning opportunities"],
                  "Medium": ["Certification and professional development support"],
                  "High": ["Advanced study and postgraduate sponsorship"]},
    "YearsAtCompany": {"Low": ["New hire integration, buddy system, and onboarding support"],
                       "Medium": ["Tenure milestone celebration and rewards"],
                       "High": ["Long-service appreciation and recognition award"]},
}

CATEGORICAL_STRATEGY_MAP = {
    "MaritalStatus": {
        "Single": ["Offer social integration and team bonding programs",
                   "Career mobility and relocation flexibility support",
                   "Mentorship pairing with senior colleagues"],
        "Married": ["Family-friendly benefits review",
                    "Flexible working hours for family commitments",
                    "Review parental leave and childcare support policies"],
        "Divorced": ["Employee assistance program (EAP) referral",
                     "Financial wellness and counselling support",
                     "Workload review to reduce additional stress"]
    },
    "Department": {
        "Cardiology": ["High-stress unit psychological support program",
                       "Critical care burnout prevention monitoring",
                       "Cardiology-specific specialist retention package"],
        "Maternity": ["Shift pattern review for maternity unit staff",
                      "Emotional resilience and support resources",
                      "Team coverage planning to reduce workload peaks"],
        "Neurology": ["Neurology specialist retention incentive",
                      "Research and publication opportunity access",
                      "Advanced neurology training sponsorship"]
    },
    "BusinessTravel": {
        "Travel_Frequently": ["Review travel frequency and workload impact",
                              "Travel compensation and allowance review",
                              "Offer remote working days after heavy travel periods"],
        "Travel_Rarely": ["Maintain current low-travel balance",
                          "Offer optional project travel for career exposure"],
        "Non-Travel": ["Offer optional travel opportunities for career growth",
                       "Virtual collaboration and networking programs"]
    },
    "EducationField": {
        "Life Sciences": ["Research grant and publication support",
                          "Life sciences CPD (continuing professional development) funding"],
        "Medical": ["Medical specialisation sponsorship",
                    "Clinical research participation opportunities"],
        "Marketing": ["Healthcare marketing career development pathway",
                      "Cross-functional project exposure"],
        "Technical Degree": ["Technical skills upgrade sponsorship",
                             "Innovation lab and project access"],
        "Human Resources": ["HR leadership development program",
                            "People analytics and strategy training"],
        "Other": ["Tailored career development discussion",
                  "Cross-disciplinary training opportunities"]
    }
}


#  INTERNAL HELPERS


def _build_quantile_models(df):
    models = {}
    for feature in NUMERIC_FEATURES:
        if feature in df.columns:
            values = pd.to_numeric(df[feature], errors="coerce")
            models[feature] = {
                "q1": values.quantile(0.33),
                "q2": values.quantile(0.66)
            }
    return models


def _get_feature_level(feature, value, models):
    model = models[feature]
    try:
        value = float(value)
    except (ValueError, TypeError):
        return "Unknown"
    if value <= model["q1"]:
        return "Low"
    elif value <= model["q2"]:
        return "Medium"
    else:
        return "High"


def _derive_role(job_role):
    return JOBROLE_GROUPS.get(job_role, "Other")


def _is_high_risk(predicted_risk):
    """Return True for High Risk and Critical Risk employees."""
    return str(predicted_risk).strip() in HIGH_RISK_LABELS


def _get_strategies_for_employee(emp_row, models, top_n=5, shap_row=None):
    """
    Generate strategies for a single employee.
    - High/Critical Risk → ROLE_BASED_STRATEGY_MAP  (corrective, role-stratified)
    - Low/Medium Risk    → LOW_RISK_STRATEGY_MAP     (preventive, not role-stratified)
    Categorical features always use CATEGORICAL_STRATEGY_MAP.
    """
    job_role = _derive_role(emp_row.get("JobRole", "Other"))
    predicted_risk = emp_row.get("Predicted_Risk", "")
    is_high = _is_high_risk(predicted_risk)
    results = []

    if shap_row:
        ranked_features = sorted(
            [(feat, val) for feat, val in shap_row.items()
             if feat in FEATURES_FOR_STRATEGY],
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_n]
    else:
        # Fallback: median deviation ranking
        feature_scores = {}
        for feature in NUMERIC_FEATURES:
            if feature in emp_row:
                try:
                    val = float(emp_row[feature])
                    med = (models[feature]["q1"] + models[feature]["q2"]) / 2
                    feature_scores[feature] = abs(val - med)
                except (ValueError, TypeError):
                    pass
        for feature in CATEGORICAL_FEATURES:
            if feature in emp_row and emp_row[feature]:
                feature_scores[feature] = 0.5
        ranked_features = sorted(
            feature_scores.items(), key=lambda x: x[1], reverse=True
        )[:top_n]

    for feature, shap_val in ranked_features:
        actual_value = emp_row.get(feature, None)
        if actual_value is None:
            continue

        if feature in models:
            level = _get_feature_level(feature, actual_value, models)

            # Choose map based on risk level
            if is_high:
                role_map = ROLE_BASED_STRATEGY_MAP.get(job_role, {})
                strategies = role_map.get(feature, {}).get(level, [])
                stype = "HighRisk-Numeric" if shap_row else "Numeric"
            else:
                strategies = LOW_RISK_STRATEGY_MAP.get(feature, {}).get(level, [])
                stype = "LowRisk-Numeric" if shap_row else "Numeric"

        else:
            # Categorical - same map for all risk levels
            strategies = CATEGORICAL_STRATEGY_MAP.get(feature, {}).get(str(actual_value), [])
            level = "N/A"
            if shap_row:
                stype = "HighRisk-Categorical" if is_high else "LowRisk-Categorical"
            else:
                stype = "Categorical"

        for strategy in strategies:
            results.append({
                "Feature": feature,
                "FeatureValue": actual_value,
                "StrategyType": stype,
                "ClusterLevel": level,
                "SHAP_Impact": shap_val if shap_row else None,
                "RecommendedStrategy": strategy
            })

    return job_role, results


#  PRIMARY FUNCTIONS (called from app.py)


def _load_combined_df():
    df = pd.read_json('jsons/combined_df_table.json')
    df['Attrition_Probability'] = (
            df['Attrition_Probability']
            .astype(str)
            .str.replace('%', '', regex=False)
            .astype(float) / 100
    )
    return df


def get_strategies(employee_dataset, top_n=5):
    """
    Generate strategies for ALL employees regardless of risk level.
    High/Critical Risk -corrective role-based strategies
    Low/Medium Risk   - preventive engagement strategies
    """
    combined_df = _load_combined_df()
    models = _build_quantile_models(combined_df)

    
    
    shap_lookup = {}
    

    all_results = []

    # ALL employees get strategies now
    for _, emp_row in combined_df.iterrows():
        emp_id = emp_row.get('EmployeeID')
        shap_row = shap_lookup.get(emp_id, None)

        role_group, strategies = _get_strategies_for_employee(
            emp_row.to_dict(), models, top_n=top_n, shap_row=shap_row
        )

        prob_col = 'RiskProbability' if 'RiskProbability' in combined_df.columns else 'Attrition_Probability'

        for s in strategies:
            all_results.append({
                "EmployeeID": emp_id,
                "Age": emp_row.get('Age'),
                "JobRole": emp_row.get('JobRole', 'Other'),
                "Department": emp_row.get('Department', ''),
                "Predicted_Risk": emp_row.get('Predicted_Risk', ''),
                "Attrition_Probability": round(float(emp_row.get(prob_col, 0)), 4),
                "Role": role_group,
                **s
            })

    result_df = pd.DataFrame(all_results)
    result_df.to_json('jsons/strategies_table.json', orient='records', indent=4)
    return result_df


def get_employee_strategy(employee_dataset, employee_id):
    """
    Generate strategies for a SINGLE employee by ID.
    Works for all risk levels.
    """
    combined_df = _load_combined_df()
    models = _build_quantile_models(combined_df)

    try:
        emp_id_int = int(employee_id)
        emp_row = combined_df[combined_df['EmployeeID'] == emp_id_int]
    except (ValueError, TypeError):
        return None

    if emp_row.empty:
        return None

    emp_row = emp_row.iloc[0].to_dict()
    predicted_risk = emp_row.get('Predicted_Risk', '')

    #  Load SHAP for this employee if available
    shap_row = None
    try:
        import feature_interpretation_be as fib
        import risk_profiling_be as rpb
        dataset_name = str(employee_dataset)[:-4] if str(employee_dataset).endswith('.csv') else str(employee_dataset)
        cleaned_df = rpb.clean_dataset(dataset_name)
        lime_values = fib.get_employee_lime_values(emp_id_int, cleaned_df)
        shap_row = {}
        for feature_desc, weight in lime_values:
            for feature_name in FEATURES_FOR_STRATEGY:
                if feature_name.lower() in feature_desc.lower():
                    shap_row[feature_name] = weight
                    break
    except Exception as e:
        print(f"[WARN] LIME unavailable for {emp_id_int}: {e}")

    role_group, strategies = _get_strategies_for_employee(
        emp_row, models, top_n=5, shap_row=shap_row
    )

    return {
        "EmployeeID": emp_id_int,
        "Age": emp_row.get('Age'),
        "JobRole": emp_row.get('JobRole'),
        "Department": emp_row.get('Department'),
        "Predicted_Risk": predicted_risk,
        "RiskCategory": "High Risk" if _is_high_risk(predicted_risk) else "Low Risk",
        "Role": role_group,
        "Strategies": strategies
    }


def get_strategy_summary(employee_dataset):
    """Return high-level summary stats for the dashboard."""
    combined_df = _load_combined_df()
    high_risk = combined_df[combined_df['Predicted_Risk'].isin(HIGH_RISK_LABELS)]
    low_risk = combined_df[combined_df['Predicted_Risk'].isin(LOW_RISK_LABELS)]

    try:
        strategies_df = pd.read_json('jsons/strategies_table.json')
        avg_strategies = round(
            len(strategies_df) / strategies_df['EmployeeID'].nunique(), 2
        ) if not strategies_df.empty else 0

        total_pool = sum(len(s) for r in ROLE_BASED_STRATEGY_MAP.values() for f in r.values() for s in f.values())
        total_pool += sum(len(s) for f in LOW_RISK_STRATEGY_MAP.values() for s in f.values())
        total_pool += sum(len(s) for f in CATEGORICAL_STRATEGY_MAP.values() for s in f.values())
        pool_coverage = round(
            strategies_df['RecommendedStrategy'].nunique() / total_pool, 2
        ) if not strategies_df.empty else 0

    except Exception:
        avg_strategies = 0
        pool_coverage = 0

    return {
        "TotalEmployees": len(combined_df),
        "HighRiskEmployees": len(high_risk),
        "LowRiskEmployees": len(low_risk),
        "CoverageScore": 1.0 if len(combined_df) > 0 else 0,
        "StrategyPoolCoverage": pool_coverage,
        "AvgStrategiesPerEmployee": avg_strategies,
        "RoleDistribution": combined_df['JobRole'].value_counts().to_dict(),
        "RiskDistribution": combined_df['Predicted_Risk'].value_counts().to_dict()
    }
