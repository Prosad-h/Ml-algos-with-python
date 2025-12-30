import pandas as pd

import matplotlib.pyplot as plt
class Kmeans:
    def fit(self,K:int,X:pd.DataFrame, epochs=1):
        self.m = X.shape[0]
        self.cols = X.shape[1]
        self.Ks = []
        for i in X.sample(n=K).values:
            self.Ks.append(i)
        self.Ks = pd.Series(self.Ks)
        self.cluster = {}
        self.cluster_df = []
        self.reset_df()
        epochs_finished = 0
        print("start")
        while epochs_finished < epochs:
            for index,row in enumerate(X.iloc):
                distances_from_centroids = []
                
                for cen_index,centroid in enumerate(self.Ks):
                    series_of_differences = centroid-row.values
                    series_of_differences = series_of_differences**2
                    sum_of_square_diff = series_of_differences.sum()
                    total_distance = sum_of_square_diff**0.5
                    total_value = (cen_index,total_distance)
                    distances_from_centroids.append(total_value)
                lowest_distance = distances_from_centroids[0]
                for index_dis,distance in distances_from_centroids:
                    
                    if distance < lowest_distance[1]:
                        lowest_distance = (index_dis,distance)
                self.cluster[index] = lowest_distance[0]
                
            self.reset_df()
            for row_index,cluster_index in enumerate(pd.Series(self.cluster)):
                
                self.cluster_df[cluster_index].loc[len(self.cluster_df[cluster_index])] = X.iloc[row_index].values
            
            for cluster_i,cluster_value in enumerate(self.Ks):
                for cluster_col_i in range(0,self.cols):
                    self.Ks[cluster_i][cluster_col_i] = self.cluster_df[cluster_i].iloc[:,cluster_col_i].mean()
                
            epochs_finished += 1
        return pd.Series(self.cluster_df)
    def reset_df(self):
        self.cluster_df = []
        
        for kluster in self.Ks:
            data = {}
            for col in range(1,self.cols+1):
                data[str(col)] = []
            self.cluster_df.append(pd.DataFrame(data))

            

df = pd.read_csv("./output.csv")
df = df.drop(columns=["color","label","batch"])


model = Kmeans()
clusters=model.fit(2,df,epochs=5)

colors =["#0000FF", "#FF0000"]
all_clusters_df = None
for i,cluster in enumerate(clusters):
    
    cluster["color"] = [colors[i]]*cluster.shape[0]
    
    if i == 0:
        
        all_clusters_df = cluster
    else:
        all_clusters_df = pd.concat([all_clusters_df,cluster], ignore_index=True)


plt.figure(1)
plt.scatter(all_clusters_df["1"], all_clusters_df["2"])
plt.title("Original Data")
plt.figure(2)
plt.scatter(all_clusters_df["1"], all_clusters_df["2"], c=all_clusters_df["color"])
plt.title("Grouped Data")
plt.show()