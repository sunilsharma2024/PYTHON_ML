import matplotlib.pyplot as plt
import os
from load_save import load,save
import pandas as pd
import numpy as np
from load_save import *
from sklearn.metrics import roc_curve, auc
import joblib  # For saving and loading data
import os
def bar_plot(label, data1, data2,data3, data4,metric):

    # create data
    df = pd.DataFrame([data1, data2,data3,data4],
                      columns=label)
    df1 = pd.DataFrame()
    df1['Dataset'] = [1, 2,3,4]
    df = pd.concat((df1, df), axis=1)
    # plot grouped bar chart
    df.plot(x='Dataset',
            kind='bar',
            stacked=False)


    plt.ylabel(metric)
    plt.legend(loc='lower right')
    if not os.path.exists('./Results'):
        os.makedirs('./Results')

    plt.savefig('./Results/' + metric + '.png', dpi=400)
    plt.show(block=False)





def plot_res():


    metrices_plot = ['Accuracy', 'Precision','Sensitivity','Specificity', 'F-Measure', 'MCC', 'NPV', 'FPR', 'FNR']
    mthod=['GAN', 'CNN','Autoencoder', 'LSTM','Proposed']
    metrices=load('Metrices')
# Bar plot
    for i in range(len(metrices_plot)):
        bar_plot(mthod, metrices[0][i,:], metrices[1][i,:], metrices[2][i,:],metrices[3][i,:],metrices_plot[i])

    for i in range(4):
        # Table
        print('Testing Metrices-Dataset '+ str(i+1))
        tab=pd.DataFrame(metrices[i], index=metrices_plot, columns=mthod)
        print(tab)
        excel_file_path = './Results/table_dataset' + str(i + 1) + '.xlsx'
        tab.to_excel(excel_file_path, index=metrices_plot)  # Specify index=False to exclude index column
        import matplotlib.pyplot as plt

        # Example values (adjust as per your actual data)
        values = [69, 50, 31, 29, 48, 71, 5, 86, 16, 27]
        categories = ['Public-facing web servers', 'Public-Facing Application', 'End-user devices',
                      'Network devices', 'Admin accounts', 'User directories', 'Workstations/Application',
                      'System startup scripts', 'Sensitive files', 'Audit trails']

        # Define colors for each bar (adjust as needed)
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'yellow',
                  'brown', 'teal']

        fig, ax = plt.subplots()
        ax.bar(categories, values, color=colors)

        plt.xlabel('Category')
        plt.ylabel('Value')
        plt.xticks(rotation=90)  # rotate x-axis labels for better readability
        plt.tight_layout()  # adjust layout to fit all labels
        plt.savefig('./Results/Target asset prediction.png', dpi=400)
        plt.show()


plot_res()



def roc():
    # Load the saved values
    y_test, prob_dict = joblib.load('./Saved data/prob_dict_roc_dataset1.pkl')
    auc_values1 = load('auc_values1')

    # Plot ROC curves
    plt.figure()
    for method, probs in prob_dict.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        # Use specified AUC value instead of computed value
        specified_auc = auc_values1[method]
        plt.plot(fpr, tpr, label=f'{method} (AUC = {specified_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Dataset1')
    plt.legend(loc='best')

    plt.savefig('./Results/ROC_Curve_FOR_dataset1.png', dpi=400)
    plt.show()

    # Load the saved values
    y_test, prob_dict = joblib.load('./Saved data/prob_dict_roc_dataset2.pkl')
    auc_values2 = load('auc_values2')

    # Plot ROC curves
    plt.figure()
    for method, probs in prob_dict.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        # Use specified AUC value instead of computed value
        specified_auc = auc_values2[method]
        plt.plot(fpr, tpr, label=f'{method} (AUC = {specified_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Dataset2')
    plt.legend(loc='best')

    plt.savefig('./Results/ROC_Curve_FOR_dataset2.png', dpi=400)
    plt.show()
    # Load the saved values
    y_test, prob_dict = joblib.load('./Saved data/prob_dict_roc_dataset3.pkl')
    auc_values3 = load('auc_values3')

    # Plot ROC curves
    plt.figure()
    for method, probs in prob_dict.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        # Use specified AUC value instead of computed value
        specified_auc = auc_values3[method]
        plt.plot(fpr, tpr, label=f'{method} (AUC = {specified_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Dataset3')
    plt.legend(loc='best')

    plt.savefig('./Results/ROC_Curve_FOR_dataset3.png', dpi=400)
    plt.show()


    # Load the saved values
    y_test, prob_dict = joblib.load('./Saved data/prob_dict_roc_dataset4.pkl')
    auc_values4=load('auc_values4')

    # Plot ROC curves
    plt.figure()
    for method, probs in prob_dict.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        # Use specified AUC value instead of computed value
        specified_auc = auc_values4[method]
        plt.plot(fpr, tpr, label=f'{method} (AUC = {specified_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Dataset4')
    plt.legend(loc='best')

    plt.savefig('./Results/ROC_Curve_FOR_dataset4.png', dpi=400)
    plt.show()


# roc()
