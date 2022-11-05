import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")


class ClusteringPrototypiqueTemporel:
    # constructor + main methods
    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.data = dataframe.copy()
        self.fitted = False

    def fit(self) -> None:
        self.initial_data = self.data.copy()

        for idx in range(self.data.shape[0]):  # Parcours de tous les time step

            # Clustering par les kmeans de sklearn
            kmeans = KMeans(n_clusters=3, algorithm="elkan", n_init=50, random_state=0)

            results = kmeans.fit(self.initial_data.iloc[idx].values.reshape(-1, 1))

            # création du dictionnaire de mapping pour associer chaque point à son cluster pour tout time step
            map_dict = dict()
            map_dict[np.argmin(results.cluster_centers_)] = "cluster_bas"
            map_dict[np.argmax(results.cluster_centers_)] = "cluster_haut"
            map_dict[
                [
                    x
                    for x in [0, 1, 2]
                    if x
                    not in [
                        np.argmin(results.cluster_centers_),
                        np.argmax(results.cluster_centers_),
                    ]
                ][0]
            ] = "cluster_moyen"

            # transformation de chaque ligne à leur cluster associé
            vectorized_func = np.vectorize(self.__mapper)
            self.data.iloc[idx] = vectorized_func(results.labels_, map_dict)

            # sauvegarde des prototypes à chaque time step ainsi que les min / max associés à chaque prototype pour chaque time step
            for prototype in ["cluster_bas", "cluster_haut", "cluster_moyen"]:
                mask = vectorized_func(results.labels_, map_dict) == prototype
                maximum = np.max(self.initial_data.iloc[0].iloc[mask])
                minimum = np.min(self.initial_data.iloc[0].iloc[mask])

                str = f"{prototype}_array"

                if idx == 0:
                    locals()[str] = np.array(
                        [
                            results.cluster_centers_[
                                self.__get_key_from_value(prototype, map_dict)
                            ][0],
                            maximum,
                            minimum,
                        ]
                    )
                else:
                    locals()[str] = np.vstack(
                        [
                            locals()[str],
                            [
                                results.cluster_centers_[
                                    self.__get_key_from_value(prototype, map_dict)
                                ][0],
                                maximum,
                                minimum,
                            ],
                        ]
                    )

        # assignation du cluster final par calcul du mode
        self.labels_ = self.data.mode(axis=0)

        # Passage des prototypes en dataframes
        self.__cluster_bas = pd.DataFrame(
            locals()["cluster_bas_array"], columns=["Moyenne", "Maximum", "Minimum"]
        )
        self.__cluster_moyen = pd.DataFrame(
            locals()["cluster_moyen_array"], columns=["Moyenne", "Maximum", "Minimum"]
        )
        self.__cluster_haut = pd.DataFrame(
            locals()["cluster_haut_array"], columns=["Moyenne", "Maximum", "Minimum"]
        )

        self.fitted = True

    def predict(self, new_client: pd.Series) -> str:
        diff_bas = np.sqrt(
            np.nansum((new_client.values - self.__cluster_bas["Moyenne"].values) ** 2)
        )
        diff_moyen = np.sqrt(
            np.nansum((new_client.values - self.__cluster_moyen["Moyenne"].values) ** 2)
        )
        diff_haut = np.sqrt(
            np.nansum((new_client.values - self.__cluster_haut["Moyenne"].values) ** 2)
        )

        if diff_haut < diff_moyen < diff_bas:
            return "Prototype haut"
        elif diff_bas < diff_moyen < diff_haut:
            return "Prototype bas"
        else:
            return "Prototype moyen"

    # getter
    def get_labels(self) -> pd.DataFrame:
        if self.fitted:
            return self.labels_.T
        raise ValueError("Modèle non fitté")

    def get_prototypes(self) -> dict:
        if self.fitted:
            return {
                "prototype_faible": self.__cluster_bas,
                "prototype_fort": self.__cluster_haut,
                "prototype_moyen": self.__cluster_moyen,
            }
        raise ValueError("Modèle non fitté")

    # plotting
    def plot_prototypes(self) -> None:
        if self.fitted:
            # Plotting des prototypes et de leur zone d'influence
            plt.figure(figsize=(14, 6))
            plt.title("Courbes prototypiques", fontsize=9, fontweight="bold")

            plt.plot(
                self.data.index,
                self.__cluster_bas["Moyenne"],
                label="Prototype bas",
                color="gold",
            )
            plt.fill_between(
                x=self.data.index,
                y1=self.__cluster_bas["Minimum"],
                y2=self.__cluster_bas["Maximum"],
                alpha=0.3,
                color="gold",
            )

            plt.plot(
                self.data.index,
                self.__cluster_moyen["Moyenne"],
                label="Prototype moyen",
                color="C0",
            )
            plt.fill_between(
                x=self.data.index,
                y1=self.__cluster_moyen["Minimum"],
                y2=self.__cluster_moyen["Maximum"],
                alpha=0.3,
                color="C0",
            )

            plt.plot(
                self.data.index,
                self.__cluster_haut["Moyenne"],
                label="Prototype haut",
                color="C3",
            )
            plt.fill_between(
                x=self.data.index,
                y1=self.__cluster_haut["Minimum"],
                y2=self.__cluster_haut["Maximum"],
                alpha=0.3,
                color="C3",
            )

            plt.legend()
            plt.show()
        else:
            raise ValueError("Modèle non fitté")

    def plot_and_predict(self, new_client: pd.Series):
        if self.fitted:
            plt.figure(figsize=(14, 6))
            plt.title("Client à clusteriser", fontsize=11, fontweight="bold")

            plt.plot(
                new_client.index,
                new_client,
                label="Client à clusteriser",
                color="black",
            )

            plt.fill_between(
                x=self.initial_data.index,
                y1=self.__cluster_bas["Minimum"],
                y2=self.__cluster_bas["Maximum"],
                alpha=0.6,
                color="gold",
                label="Zone prototype bas",
            )
            plt.fill_between(
                x=self.initial_data.index,
                y1=self.__cluster_moyen["Minimum"],
                y2=self.__cluster_moyen["Maximum"],
                alpha=0.6,
                color="C0",
                label="Zone prototype moyen",
            )
            plt.fill_between(
                x=self.initial_data.index,
                y1=self.__cluster_haut["Minimum"],
                y2=self.__cluster_haut["Maximum"],
                alpha=0.6,
                color="C3",
                label="Zone prototype haut",
            )

            plt.legend()
            plt.show()

            print(f"Prédiction : {self.predict(new_client)}")
        else:
            raise ValueError("Modèle non fitté")

    # private methods
    def __mapper(self, x: int, map_dict: dict) -> str:
        return map_dict[x]

    def __get_key_from_value(self, x: str, map_dict: dict) -> int:
        for k, v in map_dict.items():
            if v == x:
                return k
