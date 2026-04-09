class RiskAssessor:
    @staticmethod
    def calculate_risk_score(probability_positive):
        if probability_positive < 0.40:
            return "Low Risk", "green", "Biomarkers align with healthy control group."
        elif 0.40 <= probability_positive < 0.75:
            return "Medium Risk", "orange", "Subtle vocal impairments detected. Clinical follow-up recommended."
        else:
            return "High Risk", "red", "Strong indicators of Parkinsonian dysphonia detected."