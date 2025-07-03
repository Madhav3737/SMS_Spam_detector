import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def gen_pie(ham_percent):
    plt.clf()
    percentages = [ham_percent,(100-ham_percent)]
    plt.pie(percentages,explode=[0,0.1],labels=['HAM','SPAM'],autopct='%1.2f%%',colors=['green','red'],shadow=True)
    plt.title("Ham vs Spam Probability predicted by our Model")
    save_path = 'static/probab_pie_chart.png'
    plt.savefig(save_path)


