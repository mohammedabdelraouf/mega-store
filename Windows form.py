import tkinter
import preprocessing as pre
import pickle as pick
import pandas as pd
from tkinter import *
from tkinter import filedialog, messagebox

# create the main window
windows_form = Tk()
windows_form.title("Mega store")
windows_form.geometry("500x500")
windows_form['background'] = '#45809E'
color = '#45809E'


# create the file dialog for selecting the CSV file
def choose_file():
    filename = filedialog.askopenfilename(filetypes=[("Excel file", "*.csv")])
    if filename:
        path_entry.delete(0, END)
        path_entry.insert(0, filename)


# create the path entry box
label1 = Label(windows_form, text="mega-store-test-script", font=('TeX Gyre Schola', 15), border=1, bg=color)
label1.grid(pady=15, row=0, column=1)
path_label = Label(windows_form, text="CSV File Path:", bg=color)
path_label.grid(pady=15, row=1, column=0)
path_entry = Entry(windows_form, width=50)
path_entry.grid(pady=15, row=1, column=1)
path_button = Button(windows_form, text="Choose", bg='#C6AEC7', font=("Times New Roman", 12), command=choose_file)
path_button.grid(padx=10, pady=15, row=1, column=2)

# create the model selection menu

model_type = Label(windows_form, text="Select Model type:", bg=color)
model_type.grid(pady=15, row=5, column=0)
model_var1 = StringVar()
model_var1.set("")
model_type22 = OptionMenu(windows_form, model_var1, "Classification", "Regression")
model_type22.configure(bg='#FFFAFA', relief="flat", border=0.1)
model_type22.grid(pady=15, row=5, column=1)

model_var = StringVar()

model_label = Label(windows_form, text="Select Model: ", bg=color)
model_label.grid(pady=15, row=6, column=0)

data = {'Regression': ["elasticNet_model.sav", "multivariable.sav", "random_forest.sav", "poly_model.sav"],
        'Classification': ["DecisionTreeClassifier.sav", "SVMClassifier.sav", "LogisticRegression.sav",
                           "BaggingClassifier.sav", "RandomForestClassifier.sav"]}


def update_options(*args):
    models = data[model_var1.get()]
    menu = model_menu['menu']
    model_menu['menu'].delete(0, 'end')
    for model in models:
        menu.add_command(label=model, command=lambda nation=model: model_var.set(nation))


model_var1.trace('w', update_options)
model_menu = OptionMenu(windows_form, model_var, "")
model_menu.configure(bg='#FFFAFA', relief="flat", border=0.1)
model_menu.grid(pady=15, row=6, column=1)

# create the accuracy display box
accuracy_label = Label(windows_form, text="Accuracy:", bg=color)
accuracy_label.grid(pady=15, row=8, column=0)
accuracy_text = StringVar()
accuracy_text.set("")
accuracy_box = Entry(windows_form, width=20, textvariable=accuracy_text, state="readonly", bg=color, relief='groove')
accuracy_box.grid(pady=15, row=8, column=1)


# create the button for running the regression model
def run_model():
    if path_entry.get() == "":
        messagebox.showinfo("showinfo", "please select the file of data set")
        print("please select the file of data set")
        return
    e = Exception.__base__
    try:
        if model_var1.get() == "Regression":
            selected_features = pick.load(open('models/features.pkl', 'rb'))
            target_name = 'Profit'
            encoder_path = ".sav"
            statistics = pick.load(open('models/statistics1.pkl', 'rb'))
            fil_na_target = 'mean'
        elif model_var1.get() == "Classification":
            selected_features = pick.load(open('models/features2.pkl', 'rb'))
            target_name = 'ReturnCategory'
            encoder_path = "2.sav"
            statistics = pick.load(open('models/statistics2.pkl', 'rb'))
            fil_na_target = 'mode'
        else:
            messagebox.showinfo("showinfo", "please select the model type")
            print("please select the model type")
            return

        data = pd.read_csv(path_entry.get())
        Target = data.loc[:, target_name]
        features = data.drop(target_name, axis=1)
        features = pre.pre_processing(features)
        features = features[selected_features.columns]
        X_test_num, X_test_cat = pre.numerical_Categorical(features)

        # handel Na values
        skew_values = X_test_num.skew()
        for column in X_test_num.columns[X_test_num.isnull().any()]:
            skewness = skew_values[column]
            if abs(skewness) < 0.5:  # Assuming a skewness threshold of 0.5
                features[column].fillna(statistics.at[column, 'mean'], inplace=True)
            else:
                features[column].fillna(statistics.at[column, 'median'], inplace=True)

        for col in X_test_cat.columns:
            features[col].fillna(statistics.at[col, 'mode'])

        if model_var1.get() == "Classification":
            Target.fillna(statistics.at[target_name, fil_na_target])
        else:
            if abs(Target.skew()) < 0.5:  # Assuming a skewness threshold of 0.5
                Target.fillna(statistics.at[target_name, 'mean'], inplace=True)
            else:
                Target.fillna(statistics.at[target_name, 'median'], inplace=True)

        if model_var1.get() == "Classification":
            encoder1 = pick.load(open('encoders/ReturnCategory.sav', 'rb'))
            df = encoder1.inverse_transform(
                pd.Series([statistics.at['ReturnCategory', 'mode']], name='ReturnCategory', index=[0]))
            Target.fillna(df.at[0, 'ReturnCategory'])
            Target = encoder1.transform(Target)

        for col in X_test_cat:
            encoder = pick.load(open('encoders/' + col + encoder_path, 'rb'))
            df = encoder.inverse_transform(pd.Series([statistics.at[col, 'mode']], name=col, index=[0]))
            features[col].fillna(df.at[0, col])
            features.loc[:, col] = encoder.transform(features.loc[:, col])
            # ##################### #
        var = model_var.get()
        poly_features = pick.load(open("models/Polynomial.sav", "rb"))
        model = pick.load(open("models/" + var, "rb"))
        if var == "poly_model.sav":
            poly = poly_features.transform(features)
            accuracy = model.score(poly, Target)
            c = pd.Series(model.predict(poly), name="Profit")
            df = pd.DataFrame(c)
            df = pd.concat([df, Target], axis=1)
            df.columns = ['Predictions', 'actual']
            df.to_csv('output.csv', index=False, encoding='utf-8')
        elif var == "":
            print("please select the model")
            return
        else:
            if model_var1.get() == "Classification":
                encoder1 = pick.load(open('encoders/ReturnCategory.sav', 'rb'))
                bb = pd.Series(model.predict(features), name='ReturnCategory')
                c = encoder1.inverse_transform(bb)
                df = pd.DataFrame(c)
                df = pd.concat([df, encoder1.inverse_transform(Target)], axis=1)
                df.columns = ['predictions', 'actual']
            else:
                c = pd.Series(model.predict(features), name="Profit")
                df = pd.DataFrame(c)
                df = pd.concat([df, Target], axis=1)
                df.columns = ['predictions', 'actual']
            df.to_csv('output3.csv', index=False, encoding='utf-8')
            accuracy = model.score(features, Target)
        accuracy_text.set(f"{accuracy:.5f}")
        print(f"{var} Score: {accuracy:.5f}")
    except e:
        messagebox.showinfo("showinfo", "model run failed "
                                        "please try again and make sure to choose the correct model type depends "
                                        "on your dataset "
                                        "and choose the a suitable model depends on this type")

        print("model run failed"
              "please try again and make sure to choose the correct model type depends on your dataset"
              "and choose the a suitable model depends on this type")


button_border = tkinter.Frame(windows_form, highlightbackground="black",
                              highlightthickness=0.5, bd=0, relief='ridge')
run_button = Button(button_border, text="Run Model", fg='black',
                    bg='#C6AEC7', font=("Times New Roman", 14), command=run_model)
run_button.grid(row=9, column=1)

button_border.grid(pady=20, row=9, column=1)
# start the main loop
windows_form.mainloop()
