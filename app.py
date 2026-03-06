from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import f, chi2
from sklearn.metrics import confusion_matrix, roc_auc_score

app = Flask(__name__)

df=None

# ---------------- HOME ----------------

@app.route("/",methods=["GET","POST"])
def home():

    global df

    if request.method=="POST":

        file=request.files["file"]

        df=pd.read_csv(file)

        columns=df.columns.tolist()

        return render_template(
            "configure.html",
            columns=columns
        )

    return render_template("home.html")

# ---------------- RUN MODEL ----------------

@app.route("/run_model",methods=["POST"])
def run_model():

    global df

    model_type=request.form["model_type"]
    y_var=request.form["y"]
    x_vars=request.form.getlist("x")

    y=df[y_var]
    X=df[x_vars]

    X=sm.add_constant(X)

    result={}

    # -------- LINEAR REGRESSION --------

    if model_type in ["SLR","MLR"]:

        model=sm.OLS(y,X).fit()

        y_hat=model.predict(X)

        residuals=y-y_hat

        # Residual plot

        plt.figure()
        plt.scatter(y_hat,residuals)
        plt.axhline(0,color="red")
        plt.xlabel("Predicted")
        plt.ylabel("Residual")
        plt.title("Residual Plot")
        plt.savefig("static/residual.png")
        plt.close()

        # QQ plot

        sm.qqplot(residuals,line="45")
        plt.title("Normal Q-Q Plot")
        plt.savefig("static/qqplot.png")
        plt.close()

        # Equation

        params=model.params

        eq=f"{y_var} = "

        for i,c in enumerate(params):

            if i==0:
                eq+=f"{round(c,4)} "

            else:
                eq+=f"+ ({round(c,4)} × {params.index[i]}) "

        F_cal=model.fvalue
        df1=int(model.df_model)
        df2=int(model.df_resid)

        F_table=f.ppf(0.95,df1,df2)

        result={
        "type":model_type,
        "equation":eq,
        "summary":model.summary().as_html(),
        "r2":round(model.rsquared,4),
        "adj_r2":round(model.rsquared_adj,4),
        "F_cal":round(F_cal,3),
        "F_table":round(F_table,3)
        }

    # -------- LOGISTIC REGRESSION --------

    if model_type=="LOGISTIC":

        model=sm.Logit(y,X).fit()

        prob=model.predict(X)

        y_hat=(prob>0.5).astype(int)

        cm=confusion_matrix(y,y_hat)

        auc=roc_auc_score(y,prob)

        null_model=sm.Logit(y,np.ones((len(y),1))).fit(disp=0)

        LL_full=model.llf
        LL_null=null_model.llf

        LR=2*(LL_full-LL_null)

        chi_table=chi2.ppf(0.95,len(x_vars))

        params=model.params

        eq="log(p/(1-p)) = "

        for i,c in enumerate(params):

            if i==0:
                eq+=f"{round(c,4)} "

            else:
                eq+=f"+ ({round(c,4)} × {params.index[i]}) "

        result={
        "type":"LOGISTIC",
        "equation":eq,
        "summary":model.summary().as_html(),
        "LL_full":round(LL_full,3),
        "LL_null":round(LL_null,3),
        "LR":round(LR,3),
        "chi_table":round(chi_table,3),
        "confusion":cm.tolist(),
        "auc":round(auc,4)
        }

    return render_template("result.html",result=result)

if __name__=="__main__":
    app.run(debug=True)