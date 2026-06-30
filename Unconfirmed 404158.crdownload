from preprocessing import *

nb_model = MultinomialNB()

nb_model.fit(X_train, y_train)

y_pred_nb = nb_model.predict(X_test)

y_prob_nb = nb_model.predict_proba(X_test)[:, 1]

print("\n========== Naive Bayes ==========")

print("Accuracy:",
      accuracy_score(y_test, y_pred_nb))

print("Precision:",
      precision_score(y_test, y_pred_nb))

print("Recall:",
      recall_score(y_test, y_pred_nb))

print("F1 Score:",
      f1_score(y_test, y_pred_nb))

print("ROC-AUC:",
      roc_auc_score(y_test, y_prob_nb))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_nb))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_nb))

nb_cv_scores = cross_val_score(
    nb_model,
    X_tfidf,
    y,
    cv=5,
    scoring='accuracy'
)

print("\nCross Validation Scores:", nb_cv_scores)

print("Mean Accuracy:", nb_cv_scores.mean())