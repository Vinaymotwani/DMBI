import pandas as pd
import numpy as np

print("=========================================================================================")
print("                      ACADEMIC EARLY WARNING SYSTEM (EWS) STUDENT RISK REPORT             ")
print("=========================================================================================")
print(f"Total Students Analyzed: {len(X_test)}  |  Model: Decision Tree Classifier (Depth=4)")
print("Target Class: At Risk (Final Grade G3 < 10)")
print("=========================================================================================\n")

# Combine test set inputs with actuals and predictions
risk_dashboard = X_test.copy()
risk_dashboard['Actual_Status'] = y_test.map({1: 'Pass', 0: 'At Risk'})
risk_dashboard['Predicted_Status'] = y_pred_dt
risk_dashboard['Risk_Probability (%)'] = np.round(dt_classifier.predict_proba(X_test)[:, 0] * 100, 1)

# Filter for students identified as At Risk by the model
at_risk_students = risk_dashboard[risk_dashboard['Predicted_Status'] == 0].copy()

# Function to assign automated institutional intervention triggers based on decision rules
def assign_intervention(row):
    triggers = []
    if row['G2'] <= 9.5:
        triggers.append("Priority Tutoring (Low Intermediate Score)")
    if row['failures'] >= 1:
        triggers.append("Academic Counseling (History of Failures)")
    if row['absences'] >= 10:
        triggers.append("Attendance Monitoring (High Absenteeism)")
    if row['studytime'] < 2:
        triggers.append("Study Habits Workshop (<2 hrs/week)")
    return " | ".join(triggers) if triggers else "General Academic Mentorship"

at_risk_students['Recommended_Intervention'] = at_risk_students.apply(assign_intervention, axis=1)

# Display formatted action roster for advisors
report_cols = ['G1', 'G2', 'failures', 'absences', 'studytime', 'Risk_Probability (%)', 'Recommended_Intervention']
summary_roster = at_risk_students[report_cols].head(8)  # Display top 8 flagged students

print("FLAGGED AT-RISK STUDENT ROSTER FOR ACADEMIC ADVISORS (ACTION REQUIRED):")
print("-" * 105)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(summary_roster.to_string(index=True))
print("-" * 105)

# Summary Metrics for Institutional Decision-Making
total_flagged = len(at_risk_students)
correctly_flagged = sum(at_risk_students['Actual_Status'] == 'At Risk')

print("\n--- INSTITUTIONAL SUMMARY & INTERVENTION IMPACT ---")
print(f"• Total Students Flagged for Intervention: {total_flagged} out of {len(X_test)} ({total_flagged/len(X_test)*100:.1f}%)")
print(f"• Correctly Identified At-Risk Students: {correctly_flagged} (Recall Accuracy: {correctly_flagged/sum(y_test == 0)*100:.1f}%)")
print("• Recommended Next Action: Dispatch flags to Department Advisory Committee & Schedule Peer Mentors.")
print("=========================================================================================")
