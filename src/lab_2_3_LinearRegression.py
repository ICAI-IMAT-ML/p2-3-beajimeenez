# Import here whatever you may need
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class LinearRegressor:
    """
    Linear Regression model that can perform both simple and multiple linear regression.

    Attributes:
        coefficients (np.ndarray): Coefficients of the independent variables in the regression model.
        intercept (float): Intercept of the regression model.
    """

    def __init__(self):
        """Initializes the LinearRegressor model with default coefficient and intercept values."""
        self.coefficients = None
        self.intercept = None

    def fit_simple(self, X, y):
        """
        Fit the model using simple linear regression (one independent variable).

        This method calculates the coefficients for a linear relationship between
        a single predictor variable X and a response variable y.

        Args:
            X (np.ndarray): Independent variable data (1D array).
            y (np.ndarray): Dependent variable data (1D array).

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        if np.ndim(X) > 1:
            X = X.reshape(1, -1)

        # TODO: Train linear regression model with only one coefficient

        # si tenemos dos vectores X e Y , entonces la regresión lineal y = mx + b ; 
        # donde, por la teoría, sabemos que : 
        # w = Sxy / (Sx)^2 
        # covarianza = (sumatorio (xi - media_x) * yi - media_y)/ N
        # varianza = Sx^2 = sumatorio ( xi   - media_x)^2  / N
        # se irán las N al dividirse entre sí 
        media_x = np.mean(X)
        media_y = np.mean(y)
        
        cov_XY = np.sum((X - media_x) * (y - media_y))
        var_X = np.sum((X - media_x) ** 2)

        # calculamos: 
        # w = Sxy / (Sx)^2 

        self.coefficients = cov_XY / var_X

        #b = media_y - w * media_x 
        self.intercept = media_y - self.coefficients * media_x

    # This part of the model you will only need for the last part of the notebook

    def fit_multiple(self, X, y):
        """
        Fit the model using multiple linear regression (more than one independent variable).

        This method applies the matrix approach to calculate the coefficients for
        multiple linear regression.

        Args:
            X (np.ndarray): Independent variable data (2D array where each column is a variable).
            y (np.ndarray): Dependent variable data (1D array).

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        # TODO: Train linear regression model with multiple coefficients

        # ahora, tenemos que operar con matrices. sabemos, por la teoria, que 
        # y = bo + b1 x1 + b2 x2 + ... + bn xn 
        # w = (Xt * X)^(-1) * Xt * y 
        # donde bo será el primer término de w 

        # añadimos una fila de unos a X
        fila_unos = np.ones((X.shape[0], 1)) 
        X_completa = np.hstack((fila_unos, X))
        #y_completa = np.append(1, y)
        #calculamso el verctor w = (bo, b1, b2, ...)

        Xt_X_inv =  np.linalg.inv(np.dot(np.transpose(X_completa), X_completa))
        Xt_X_inv_Xt = np.dot (Xt_X_inv , np.transpose(X_completa))
        resultados  = np.dot (Xt_X_inv_Xt, y)
        #tomamos los coeficientes (b1, .., bn)
        self.coefficients = resultados[1:]
        # tomamos el bo como el intercept 
        self.intercept = resultados[0]

    def predict(self, X): 
        """
        Predict the dependent variable values using the fitted model.

        Args:
            X (np.ndarray): Independent variable data (1D or 2D array).

        Returns:
            np.ndarray: Predicted values of the dependent variable.

        Raises:
            ValueError: If the model is not yet fitted.
        """

       
        if self.coefficients is None or self.intercept is None:
            raise ValueError("Model is not yet fitted")
        X = np.asarray(X)
        w = self.coefficients
        b = self.intercept
        if np.ndim(X) == 1:
            # TODO: Predict when X is only one variable
            # y = wx + b 

            # x lo convertimos en un array de 2D con una columna 
            X = X.reshape(-1, 1)

            # si w es un solo numero, lo convertios en un array: 
            if w.ndim == 0:
                w = np.array([w])

        else: 
       
            # TODO: Predict when X is more than one variable 
            # calculamos y = b + x * w cuando X es una matriz 2D 
            # convertimos la w en una matriz para poder hacer el calculo. 

            w = np.asarray(w)
            if w.ndim == 1:
                w = w.reshape(-1, 1)
            
        predictions =  b + X @ w

        return predictions.flatten()


def evaluate_regression(y_true, y_pred):
    """
    Evaluates the performance of a regression model by calculating R^2, RMSE, and MAE.

    Args:
        y_true (np.ndarray): True values of the dependent variable.
        y_pred (np.ndarray): Predicted values by the regression model.

    Returns:
        dict: A dictionary containing the R^2, RMSE, and MAE values.
    """
    # R^2 Score
    # TODO: Calculate R^2

    # R^2 = 1 - RSS/TSS 

    RSS = np.sum((y_true-y_pred)**2)
    TSS = np.sum((y_true - np.mean(y_true))**2)
    r_squared = 1 - (RSS/TSS)

    # Root Mean Squared Error
    # TODO: Calculate RMSE -> raiz_cuadrada ( sumatorio ( y_real - y_pred)**2 )
    N = len(y_true)

    rmse = np.sqrt(np.sum((y_true - y_pred)**2)/N) 

    # Mean Absolute Error
    # TODO: Calculate MAE
    mae = np.sum(abs(y_true - y_pred))/N

    return {"R2": r_squared, "RMSE": rmse, "MAE": mae}


# ### Scikit-Learn comparison


def sklearn_comparison(x, y, linreg):
    """Compares a custom linear regression model with scikit-learn's LinearRegression.

    Args:
        x (numpy.ndarray): The input feature data (1D array).
        y (numpy.ndarray): The target values (1D array).
        linreg (object): An instance of a custom linear regression model. Must have
            attributes `coefficients` and `intercept`.

    Returns:
        dict: A dictionary containing the coefficients and intercepts of both the
            custom model and the scikit-learn model. Keys are:
            - "custom_coefficient": Coefficient of the custom model.
            - "custom_intercept": Intercept of the custom model.
            - "sklearn_coefficient": Coefficient of the scikit-learn model.
            - "sklearn_intercept": Intercept of the scikit-learn model.
    """
    ### Compare your model with sklearn linear regression model
    # TODO : Import Linear regression from sklearn
    from sklearn.linear_model import LinearRegression
    
    # Assuming your data is stored in x and y
    # TODO : Reshape x to be a 2D array, as scikit-learn expects 2D inputs for the features
    # hacemos del vector una matriz 
    x_reshaped = np.reshape(x, (-1,1))


    # Create and train the scikit-learn model
    # TODO : Train the LinearRegression model


    sklearn_model = LinearRegression()
    sklearn_model.fit(x_reshaped, y)

    # Now, you can compare coefficients and intercepts between your model and scikit-learn's model
    print("Custom Model Coefficient:", linreg.coefficients)
    print("Custom Model Intercept:", linreg.intercept)
    print("Scikit-Learn Coefficient:", sklearn_model.coef_[0])
    print("Scikit-Learn Intercept:", sklearn_model.intercept_)
    return {
        "custom_coefficient": linreg.coefficients,
        "custom_intercept": linreg.intercept,
        "sklearn_coefficient": sklearn_model.coef_[0],
        "sklearn_intercept": sklearn_model.intercept_,
    }

def anscombe_quartet():
    """Loads Anscombe's quartet, fits custom linear regression models, and evaluates performance.

    Returns:
        tuple: A tuple containing:
            - anscombe (pandas.DataFrame): The Anscombe's quartet dataset.
            - datasets (list): A list of unique dataset identifiers in Anscombe's quartet.
            - models (dict): A dictionary where keys are dataset identifiers and values
              are the fitted custom linear regression models.
            - results (dict): A dictionary containing evaluation metrics (R2, RMSE, MAE)
              for each dataset.
    """
    # Load Anscombe's quartet
    # These four datasets are the same as in slide 19 of chapter 02-03: Linear and logistic regression
    anscombe = sns.load_dataset("anscombe") #data frame 

    # Anscombe's quartet consists of four datasets
    # TODO: Construct an array that contains, for each entry, the identifier of each dataset
    
    datasets =  np.unique(anscombe["dataset"])
    # datasets = [I; II; III; IV], devuelve los 4 tipos simplemente 

    models = {}
    results = {"R2": [], "RMSE": [], "MAE": []}
    for dataset in datasets:

        # Filter the data for the current dataset
        # TODO
        # filtramos el df por los datos que tienen "I", "II", ... 
        data =anscombe [anscombe["dataset"] == dataset ]

        # Create a linear regression model
        # TODO
        model = LinearRegressor()
        
        # Fit the model
        # TODO
        X = np.array(data["x"] ) # Predictor, make it 1D for your custom model
        y = np.array(data["y"])  # Response
        model.fit_simple(X, y)

        # Create predictions for dataset
        # TODO
        y_pred = model.predict(X)

        # Store the model for later use
        models[dataset] = model

        # Print coefficients for each dataset
        print(
            f"Dataset {dataset}: Coefficient: {model.coefficients}, Intercept: {model.intercept}"
        )

        evaluation_metrics = evaluate_regression(y, y_pred)

        # Print evaluation metrics for each dataset
        print(
            f"R2: {evaluation_metrics['R2']}, RMSE: {evaluation_metrics['RMSE']}, MAE: {evaluation_metrics['MAE']}"
        )
        results["R2"].append(evaluation_metrics["R2"])
        results["RMSE"].append(evaluation_metrics["RMSE"])
        results["MAE"].append(evaluation_metrics["MAE"])
    return anscombe, datasets, models, results
    # return results


# Go to the notebook to visualize the results
