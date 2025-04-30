# APC
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns

# Charger le jeu de données
path = r"C:/Users/sadal/OneDrive/Desktop/dataPT2.csv"
df = pd.read_csv(path, sep=";", dtype={'ColumnName': 'str'}, low_memory=False)

# Fonction pour sélectionner uniquement les variables quantitatives
def get_quantitative_columns(df):
    return df.select_dtypes(include=[np.number])

# Récupérer uniquement les variables quantitatives
df_numeric = get_quantitative_columns(df)

# Garde uniquement les 25 premières variables quantitatives
df_numeric = df_numeric.iloc[:, :20]

# Gérer les valeurs manquantes
df_numeric = df_numeric.fillna(df_numeric.mean())

# Affichage de la matrice de corrélation
plt.figure(figsize=(12, 10))
sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Matrice de Corrélation")
plt.show()

# Standardisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

# ACP (Analyse en Composantes Principales)
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Sélectionner le nombre optimal de composantes (90% de variance expliquée)
cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
n_components = np.argmax(cumsum_variance >= 0.90) + 1

# Réduction de dimension
pca = PCA(n_components=n_components)
X_pca_reduced = pca.fit_transform(X_scaled)
print(f"Nombre de dimensions choisies : {n_components}")

# Visualisation de la variance expliquée
plt.plot(cumsum_variance)
plt.xlabel("Nombre de composantes")
plt.ylabel("Variance expliquée cumulée")
plt.title("Variance expliquée par l'ACP")
plt.show()

# Trouver les k plus proches voisins
def find_knn(data, k=5):
    knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    knn.fit(data)
    distances, indices = knn.kneighbors(data)
    return distances, indices

k = 5
distances, indices = find_knn(X_pca_reduced, k)
print("Distances des 5 premiers points avec leurs voisins :")
print(pd.DataFrame(distances[:5], columns=[f"Voisin {i+1}" for i in range(k)]))

# Génération de points synthétiques par mélange des voisins
def generate_avatar(data, indices):
    avatars = []
    for neighbors in indices:
        weights = np.random.exponential(scale=1.0, size=len(neighbors))
        weights /= np.sum(weights)
        avatar = np.sum(weights[:, None] * data[neighbors], axis=0)
        avatars.append(avatar)
    return np.array(avatars)

avatars = generate_avatar(X_pca_reduced, indices)

# Reconstruction des données
X_reconstructed = pca.inverse_transform(avatars)

# Conversion en DataFrame
df_synthetic = pd.DataFrame(X_reconstructed, columns=df_numeric.columns)

# **Réduction de la taille du fichier**
# 1️⃣ Convertir en float32 pour économiser de la mémoire
df_synthetic = df_synthetic.astype(np.float32)

# 2️⃣ Arrondir les valeurs à 4 décimales pour réduire encore la taille
df_synthetic = df_synthetic.round(4)

# 3️⃣ Compression avec XZ (meilleur que GZIP/BZ2 pour un CSV)
compressed_path = r"C:/Users/sadal/OneDrive/Desktop/synthetic_data.csv.xz"
df_synthetic.to_csv(compressed_path, sep=";", index=False, compression='xz')

# 4️⃣ Alternative : Enregistrer en format Parquet (encore plus petit)
parquet_path = r"C:/Users/sadal/OneDrive/Desktop/synthetic_data.parquet"
df_synthetic.to_parquet(parquet_path, engine="pyarrow", compression="snappy")

print(f"Données synthétiques enregistrées sous {compressed_path}")
print(f"Données synthétiques également disponibles en Parquet sous {parquet_path}")
