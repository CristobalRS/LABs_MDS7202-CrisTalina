def get_best_model(experiment_id):
    runs = mlflow.search_runs(experiment_id)
    best_model_id = runs.sort_values("metrics.valid_f1")["run_id"].iloc[0]
    best_model = mlflow.sklearn.load_model("runs:/" + best_model_id + "/model")

    return best_model

optuna.logging.set_verbosity(optuna.logging.ERROR)

# División de datos
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

#nombre del experimento
experiment_name = "XGBoost"

# créar el experimento 
mlflow.create_experiment(experiment_name)

def objective_function(trial):
    # Hiperparámetros 
    params = {
        "objective": "multi:softmax",
        "eval_metric": "mlogloss",
        "num_class": 2,
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.001, 0.1),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
        "gamma": trial.suggest_float("gamma", 0, 1),
        "n_estimators": trial.suggest_int("n_estimators", 10, 300),
    }

    # Entrenamiento y evaluación del modelo
    model = XGBClassifier(seed=42, **params)
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], verbose=False)
    yhat = model.predict(X_valid)
    valid_f1 = f1_score(y_valid, yhat, average="weighted")

    # Registrar cada entrenamiento en un experimento nuevo,
    run_name = f"XGBoost_lr_{params['learning_rate']:.3f}_md_{params['max_depth']}"
    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.log_params(params)
        mlflow.log_metric("valid_f1", valid_f1)
        
        mlflow.xgboost.log_model(model, artifact_path="model")
    return valid_f1

def optimize_model():
    experiment = mlflow.get_experiment_by_name("XGBoost")
    experiment_id = experiment.experiment_id
    
    # Optimizar los hiperparámetros del modelo XGBoost usando Optuna.
    with mlflow.start_run(run_name="Opt_Optuna"):
        
        study = optuna.create_study(direction="maximize")
        study.optimize(objective_function, n_trials=50)  
        
        mlflow.log_params(study.best_params)
        mlflow.log_metric("valid_f1", study.best_value)

        #Devolver el mejor modelo 
        best_model = get_best_model(experiment_id)
    
        # Serializar el modelo 
        with open('optuna_model.pkl', 'wb') as file:
            pickle.dump(best_model, file)
        
        mlflow.log_artifact("optuna_model.pkl", artifact_path="models")
        
        #Guardar los gráficos de Optuna
        plots_dir = "plots"
        os.makedirs(plots_dir, exist_ok=True)

        fig_history = plot_optimization_history(study)
        fig_history.write_image(f"{plots_dir}/optimization_history.png")

        fig_importances = plot_param_importances(study)
        fig_importances.write_image(f"{plots_dir}/param_importances.png")

        #Respalde las configuraciones del modelo final y la importancia de las variables
        feature_names = X_train.columns if hasattr(X_train, 'columns') else [f"Feature {i}" for i in range(X_train.shape[1])]
        importance_plot_path = f"{plots_dir}/feature_importances.png"
        save_feature_importances(best_model, feature_names, importance_plot_path)

        mlflow.log_artifacts(plots_dir, artifact_path="plots")

if __name__ == "__main__":
    optimize_model()