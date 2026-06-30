from preprocessing import *

log_model = LogisticRegression(max_iter=1000)

log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)

y_prob_log = log_model.predict_proba(X_test)[:, 1]

print("\n========== Logistic Regression ==========")

print("Accuracy:",
      accuracy_score(y_test, y_pred_log))

print("Precision:",
      precision_score(y_test, y_pred_log))

print("Recall:",
      recall_score(y_test, y_pred_log))

print("F1 Score:",
      f1_score(y_test, y_pred_log))

print("ROC-AUC:",
      roc_auc_score(y_test, y_prob_log))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_log))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_log))

log_cv_scores = cross_val_score(
    log_model,
    X_tfidf,
    y,
    cv=5,
    scoring='accuracy'
)

print("\nCross Validation Scores:", log_cv_scores)

print("Mean Accuracy:", log_cv_scores.mean())