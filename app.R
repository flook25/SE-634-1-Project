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

ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE, savePredictions = TRUE)

# Train models
logit_model <- train(diabetes ~ .,
                     data = train_df,
                     method = "glm",
                     family = "binomial",
                     trControl = ctrl)
logit_probs_val <- predict(logit_model, newdata = val_df, type = "prob")
logit_probs_test <- predict(logit_model, newdata = test_df, type = "prob")

tree_model <- train(diabetes ~ .,
                    data = train_df,
                    method = "rpart",
                    trControl = ctrl)
tree_probs_val <- predict(tree_model, newdata = val_df, type = "prob")
tree_probs_test <- predict(tree_model, newdata = test_df, type = "prob")

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
knn_probs_val <- predict(knn_model, newdata = val_knn, type = "prob")
knn_probs_test <- predict(knn_model, newdata = test_knn, type = "prob")

# UI
ui <- fluidPage(
  theme = shinytheme("cosmo"),
  tags$head(tags$style(HTML("
    .title-logo { display: flex; align-items: center; gap: 15px; }
    .title-logo img { height: 50px; }
    .pred-box { padding: 20px; border-radius: 10px; margin-bottom: 20px; }
    .pred-neg { background: #e6ffe6; }
    .pred-pos { background: #ffe6e6; }
    .prob-bar { height: 20px; border-radius: 10px; margin: 3px 0; background: #f0f0f0; }
    .prob-fill-neg { background: #4caf50; height: 100%; border-radius: 10px; }
    .prob-fill-pos { background: #f44336; height: 100%; border-radius: 10px; }
  "))),
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
      tabsetPanel(
        tabPanel("Prediction",
                 uiOutput("prediction_output"),
                 plotOutput("roc_plot")
        ),
        tabPanel("Model Performance",
                 radioButtons("data_set", "Select Dataset:", 
                              choices = c("Validation", "Test"), 
                              selected = "Validation", 
                              inline = TRUE),
                 plotOutput("metric_plot")
        ),
        tabPanel("Details",
                 verbatimTextOutput("val_output"),
                 verbatimTextOutput("test_output")
        )
      )
    )
  )
)

# Server
server <- function(input, output, session) {
  
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
  
  # Enhanced prediction output
  observeEvent(input$predict_btn, {
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
    
    pred_class <- switch(input$model_type,
                         "Logistic Regression" = predict(logit_model, newdata = newdata),
                         "Decision Tree" = predict(tree_model, newdata = newdata),
                         "KNN" = {
                           input_norm <- predict(preproc, newdata)
                           predict(knn_model, newdata = input_norm)
                         })
    pred_prob <- switch(input$model_type,
                        "Logistic Regression" = predict(logit_model, newdata = newdata, type = "prob"),
                        "Decision Tree" = predict(tree_model, newdata = newdata, type = "prob"),
                        "KNN" = {
                          input_norm <- predict(preproc, newdata)
                          predict(knn_model, newdata = input_norm, type = "prob")
                        })
    prob_pos <- round(pred_prob$pos * 100, 1)
    prob_neg <- round(pred_prob$neg * 100, 1)
    
    # Color and emoji logic
    if(pred_class == "pos") {
      box_class <- "pred-box pred-pos"
      emoji <- if(prob_pos > 90) "⚠️" else if(prob_pos > 70) "🔴" else "🟠"
      headline <- paste("Prediction: Positive", emoji)
    } else {
      box_class <- "pred-box pred-neg"
      emoji <- if(prob_neg > 90) "✅" else if(prob_neg > 70) "🟢" else "🟡"
      headline <- paste("Prediction: Negative", emoji)
    }
    
    output$prediction_output <- renderUI({
      tagList(
        div(class = box_class,
            h3(headline),
            p(strong("Model:"), input$model_type),
            div(
              "Probability:",
              div(style = "display:flex; align-items:center;",
                  div(style = "width:80px;", "Negative:"),
                  div(class = "prob-bar", style = "flex:1; margin-right:10px;",
                      div(class = "prob-fill-neg", style = sprintf("width:%s%%;", prob_neg))
                  ),
                  div(style = "width:50px; text-align:right;", paste0(prob_neg, "%"))
              ),
              div(style = "display:flex; align-items:center;",
                  div(style = "width:80px;", "Positive:"),
                  div(class = "prob-bar", style = "flex:1; margin-right:10px;",
                      div(class = "prob-fill-pos", style = sprintf("width:%s%%;", prob_pos))
                  ),
                  div(style = "width:50px; text-align:right;", paste0(prob_pos, "%"))
              )
            )
        )
      )
    })
  })
  
  # ROC plot (validation set only for simplicity)
  output$roc_plot <- renderPlot({
    actual_numeric <- ifelse(val_df$diabetes == "pos", 1, 0)
    roc_logit <- roc(actual_numeric, logit_probs_val$pos)
    roc_tree <- roc(actual_numeric, tree_probs_val$pos)
    roc_knn <- roc(actual_numeric, knn_probs_val$pos)
    
    plot(roc_logit, col = "blue", legacy.axes = TRUE, main = "ROC Curves for Validation Set")
    lines(roc_tree, col = "green")
    lines(roc_knn, col = "red")
    legend("bottomright", legend = c(
      paste("Logistic (AUC =", round(auc(roc_logit), 2), ")"),
      paste("Tree (AUC =", round(auc(roc_tree), 2), ")"),
      paste("KNN (AUC =", round(auc(roc_knn), 2), ")")
    ), col = c("blue", "green", "red"), lwd = 2)
  })
  
  # Performance metrics plot with toggle
  output$metric_plot <- renderPlot({
    dset <- input$data_set
    if (dset == "Validation") {
      df <- val_df
      df_knn <- val_knn
    } else {
      df <- test_df
      df_knn <- test_knn
    }
    models <- c("Decision Tree", "KNN", "Logistic Regression")
    acc <- c(
      confusionMatrix(predict(tree_model, df), df$diabetes)$overall["Accuracy"],
      confusionMatrix(predict(knn_model, df_knn), df_knn$diabetes)$overall["Accuracy"],
      confusionMatrix(predict(logit_model, df), df$diabetes)$overall["Accuracy"]
    )
    sensitivity <- c(
      confusionMatrix(predict(tree_model, df), df$diabetes)$byClass["Sensitivity"],
      confusionMatrix(predict(knn_model, df_knn), df_knn$diabetes)$byClass["Sensitivity"],
      confusionMatrix(predict(logit_model, df), df$diabetes)$byClass["Sensitivity"]
    )
    specificity <- c(
      confusionMatrix(predict(tree_model, df), df$diabetes)$byClass["Specificity"],
      confusionMatrix(predict(knn_model, df_knn), df_knn$diabetes)$byClass["Specificity"],
      confusionMatrix(predict(logit_model, df), df$diabetes)$byClass["Specificity"]
    )
    df_metrics <- data.frame(
      Model = rep(models, each = 3),
      Metric = rep(c("Accuracy", "Sensitivity", "Specificity"), times = 3),
      Value = c(acc, sensitivity, specificity)
    )
    ggplot(df_metrics, aes(x = Model, y = Value, fill = Metric)) +
      geom_bar(stat = "identity", position = "dodge") +
      geom_text(aes(label = round(Value, 2)), 
                position = position_dodge(width = 0.9), 
                vjust = -0.5, size = 4) +
      labs(title = paste("Model Performance on", dset, "Set"), y = "Score", x = "") +
      scale_fill_brewer(palette = "Set2") +
      theme_minimal() +
      ylim(0, 1.1)
  })
}

# Run the app
shinyApp(ui = ui, server = server)
