
# coding: utf-8

# In[2]:


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk


# In[5]:


X = np.load('X.npy')
movie_names = np.load('movie_names.npy')
movie_size = len(X)


# In[103]:


def get_recommend():
    count = 3
    movie = entry1.get()
    for label in win.grid_slaves():
        if int(label.grid_info()['row']) > 2:
            label.grid_forget()
    if movie not in movie_names:
        label = tk.Label(win, text="找不到這部電影", font=('Arial', 12))
        label.grid(row=count, column=1, sticky=tk.W)
        return
    movie_id = np.where(movie_names == movie)[0][0]
    movie_vec = X[movie_id]
    scores = []
    for i in range(movie_size):
        score = cosine_similarity(movie_vec.reshape(1, -1), X[i].reshape(1, -1))
        scores.append(score[0])
    rank_names = np.argsort(np.array(scores).flatten())
    rank_scores = np.sort(np.array(scores).flatten())
    for name in movie_names[rank_names[-6:-1]]:
        label = tk.Label(win, text=name, font=('Arial', 12))
        label.grid(row=count, column=1, sticky=tk.W)
        count += 1
    return


# In[104]:


win = tk.Tk()
win.title('電影推薦系統')
win.geometry('300x300+500+300')
label1 = tk.Label(win, text="電影名稱：", font=('Arial', 12), height=2) 
label1.grid(row=0, column=0, padx=2, pady=15)
entry1 = tk.Entry(win)
entry1.grid(row=0, column=1, padx=5, pady=15)
movie = entry1.get()
button = tk.Button(win, text='Search',
                   command=get_recommend).grid(row=1, column=1, sticky=tk.E)
label1 = tk.Label(win, text="推薦結果：", font=('Arial', 12), width=12, height=2) 
label1.grid(row=2, column=0, padx=5, pady=5)
win.mainloop()

