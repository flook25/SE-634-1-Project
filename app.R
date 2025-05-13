# Load libraries
library(shiny)
library(caret)
library(mlbench)
library(rpart)
library(dplyr)
library(class)
library(ggplot2)
library(pROC)
library(shinythemes)

# Load and prepare data
data(PimaIndiansDiabetes)
df <- PimaIndiansDiabetes

set.seed(42)
n <- nrow(df)
train_idx <- sample(1:n, size = 0.7 * n)
remaining <- setdiff(1:n, train_idx)
val_idx <- sample(remaining, size = 0.5 * length(remaining))
test_idx <- setdiff(remaining, val_idx)

train_df <- df[train_idx, ]
val_df <- df[val_idx, ]
test_df <- df[test_idx, ]

ctrl <- trainControl(method = "cv", number = 5)

# Train models
logit_model <- train(diabetes ~ . - triceps - insulin,
                     data = train_df,
                     method = "glm",
                     family = "binomial",
                     trControl = ctrl)
logit_probs <- predict(logit_model, newdata = val_df, type = "prob")

tree_model <- train(diabetes ~ .,
                    data = train_df,
                    method = "rpart",
                    trControl = ctrl)
tree_probs <- predict(tree_model, newdata = val_df, type = "prob")

preproc <- preProcess(train_df[, -9], method = c("center", "scale"))
train_knn <- predict(preproc, train_df[, -9])
val_knn <- predict(preproc, val_df[, -9])
test_knn <- predict(preproc, test_df[, -9])
train_knn$diabetes <- train_df$diabetes
val_knn$diabetes <- val_df$diabetes
test_knn$diabetes <- test_df$diabetes

knn_model <- train(diabetes ~ ., 
                   data = train_knn,
                   method = "knn",
                   tuneLength = 5,
                   trControl = ctrl)
knn_probs <- predict(knn_model, newdata = val_knn, type = "prob")

# UI
ui <- fluidPage(
  theme = shinytheme("cosmo"),
  tags$head(tags$style(".title-logo { display: flex; align-items: center; gap: 15px; } .title-logo img { height: 50px; }")),
  div(class = "title-logo",
      img(src = "https://admissions.siit.tu.ac.th/wp-content/uploads/2023/06/cropped-TU-SIIT1992-01.png"),
      h3("Term Project - SE 634-1: AI & Data Analytics in Supply Chain Systems"),
      h4("Advisor: Dr. Warrut Pannakkong")
  ),
  titlePanel("Diabetes Prediction and Model Comparison"),
  sidebarLayout(
    sidebarPanel(
      selectInput("model_type", "Select Model:",
                  choices = c("Logistic Regression", "Decision Tree", "KNN")),
      h4("Enter Values for Prediction"),
      numericInput("pregnant", "Pregnancies:", 1),
      numericInput("glucose", "Glucose:", 100),
      numericInput("pressure", "Blood Pressure:", 70),
      numericInput("triceps", "Skin Thickness:", 20),
      numericInput("insulin", "Insulin:", 80),
      numericInput("mass", "BMI:", 30),
      numericInput("pedigree", "Diabetes Pedigree:", 0.5),
      numericInput("age", "Age:", 33),
      actionButton("predict_btn", "Predict")
    ),
    mainPanel(
      verbatimTextOutput("val_output"),
      verbatimTextOutput("test_output"),
      verbatimTextOutput("prediction_output"),
      plotOutput("roc_plot"),
      plotOutput("metric_plot")
    )
  )
)

# Server
server <- function(input, output) {
  
  val_pred <- reactive({
    switch(input$model_type,
           "Logistic Regression" = predict(logit_model, newdata = val_df),
           "Decision Tree" = predict(tree_model, newdata = val_df),
           "KNN" = predict(knn_model, newdata = val_knn))
  })
  
  test_pred <- reactive({
    switch(input$model_type,
           "Logistic Regression" = predict(logit_model, newdata = test_df),
           "Decision Tree" = predict(tree_model, newdata = test_df),
           "KNN" = predict(knn_model, newdata = test_knn))
  })
  
  output$val_output <- renderPrint({
    cat("Validation Performance (", input$model_type, ")\n")
    pred <- val_pred()
    true <- if (input$model_type == "KNN") val_knn$diabetes else val_df$diabetes
    print(confusionMatrix(pred, true))
  })
  
  output$test_output <- renderPrint({
    cat("\nTest Performance (", input$model_type, ")\n")
    pred <- test_pred()
    true <- if (input$model_type == "KNN") test_knn$diabetes else test_df$diabetes
    print(confusionMatrix(pred, true))
  })
  
  observeEvent(input$predict_btn, {
    # Prepare input data for prediction
    newdata <- data.frame(
      pregnant = input$pregnant,
      glucose = input$glucose,
      pressure = input$pressure,
      triceps = input$triceps,
      insulin = input$insulin,
      mass = input$mass,
      pedigree = input$pedigree,
      age = input$age
    )
    
    pred <- switch(input$model_type,
                   "Logistic Regression" = predict(logit_model, newdata = newdata),
                   "Decision Tree" = predict(tree_model, newdata = newdata),
                   "KNN" = {
                     input_norm <- predict(preproc, newdata)
                     predict(knn_model, newdata = input_norm)
                   })
    
    output$prediction_output <- renderPrint({
      cat("\nPrediction for input (", input$model_type, "): ", as.character(pred), "\n")
    })
  })
  
  output$roc_plot <- renderPlot({
    actual_numeric <- ifelse(val_df$diabetes == "pos", 1, 0)
    roc_logit <- roc(actual_numeric, logit_probs$pos)
    roc_tree <- roc(actual_numeric, tree_probs$pos)
    roc_knn <- roc(actual_numeric, knn_probs$pos)
    
    plot(roc_logit, col = "blue", legacy.axes = TRUE, main = "ROC Curves for Validation Set")
    lines(roc_tree, col = "green")
    lines(roc_knn, col = "red")
    legend("bottomright", legend = c(
      paste("Logistic (AUC =", round(auc(roc_logit), 2), ")"),
      paste("Tree (AUC =", round(auc(roc_tree), 2), ")"),
      paste("KNN (AUC =", round(auc(roc_knn), 2), ")")
    ), col = c("blue", "green", "red"), lwd = 2)
  })
  
  output$metric_plot <- renderPlot({
    models <- c("Logistic Regression", "Decision Tree", "KNN")
    acc <- c(
      confusionMatrix(predict(logit_model, val_df), val_df$diabetes)$overall["Accuracy"],
      confusionMatrix(predict(tree_model, val_df), val_df$diabetes)$overall["Accuracy"],
      confusionMatrix(predict(knn_model, val_knn), val_knn$diabetes)$overall["Accuracy"]
    )
    
    sensitivity <- c(
      confusionMatrix(predict(logit_model, val_df), val_df$diabetes)$byClass["Sensitivity"],
      confusionMatrix(predict(tree_model, val_df), val_df$diabetes)$byClass["Sensitivity"],
      confusionMatrix(predict(knn_model, val_knn), val_knn$diabetes)$byClass["Sensitivity"]
    )
    
    specificity <- c(
      confusionMatrix(predict(logit_model, val_df), val_df$diabetes)$byClass["Specificity"],
      confusionMatrix(predict(tree_model, val_df), val_df$diabetes)$byClass["Specificity"],
      confusionMatrix(predict(knn_model, val_knn), val_knn$diabetes)$byClass["Specificity"]
    )
    
    df_metrics <- data.frame(
      Model = rep(models, each = 3),
      Metric = rep(c("Accuracy", "Sensitivity", "Specificity"), times = 3),
      Value = c(acc, sensitivity, specificity)
    )
    
    ggplot(df_metrics, aes(x = Model, y = Value, fill = Metric)) +
      geom_bar(stat = "identity", position = "dodge") +
      labs(title = "Model Performance on Validation Set", y = "Score", x = "") +
      scale_fill_brewer(palette = "Set2") +
      theme_minimal()
  })
}

# Run the app
shinyApp(ui = ui, server = server)
