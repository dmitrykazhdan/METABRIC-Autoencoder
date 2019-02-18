import pandas
import numpy as np
import scipy
from scipy import stats
from numpy import median
from keras.layers import Input, Dense, Concatenate
from keras.models import Model
from keras import optimizers
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier


data_prefix = ""
model_path = ""


cna_data_path = data_prefix + "data_CNA.txt"
rna_data_path = data_prefix + "data_RNA_Seq_expression_median.txt"
patient_data_path = data_prefix + "data_clinical_patient.txt"
model_name = model_path + "my_model"


# Compute entropy for CNA variables
def entropy(x):
    unique, counts = np.unique(x, return_counts=True)
    counts = counts / sum(counts)
    return scipy.stats.entropy(counts)


# Compute Median Absolute Deviation for RNA variables
def MAD(x):
    return median(abs(x - median(x)))


# Group CNA variables
def normalize_cna(x):
    if x == -1 or x == -2:
        x = -1
    elif x == 1 or x == 2:
        x = 1
    else:
        x = 0
    return x


# Define the set of PAM50 genes
PAM50_genes = ['FOXC1', 'MIA', 'KNTC2', 'CEP55', 'ANLN',
               'MELK', 'GPR160', 'TMEM45B',
               'ESR1', 'FOXA1', 'ERBB2', 'GRB7',
               'FGFR4', 'BLVRA', 'BAG1', 'CDC20',
               'CCNE1', 'ACTR3B', 'MYC', 'SFRP1',
               'KRT17', 'KRT5', 'MLPH', 'CCNB1', 'CDC6',
               'TYMS', 'UBE2T', 'RRM2', 'MMP11',
               'CXXC5', 'ORC6L', 'MDM2', 'KIF2C', 'PGR',
               'MKI67', 'BCL2', 'EGFR', 'PHGDH',
               'CDH3', 'NAT1', 'SLC39A6',
               'MAPT', 'UBE2C', 'PTTG1', 'EXO1', 'CENPF',
               'CDCA1', 'MYBL2', 'BIRC5']

data = []


def train_graph():

    # Load patient data from file
    patient_data = pandas.read_csv(patient_data_path, sep="\t", skiprows=[0, 1, 2, 3])
    intclust_data = patient_data[['PATIENT_ID', 'INTCLUST']].dropna()

    # Load CNA data from file
    cna_data = pandas.read_csv(cna_data_path, sep="\t").dropna()
    cna_data = cna_data.drop(['Entrez_Gene_Id'], axis=1)

    # Load RNA data from file
    rna_data = pandas.read_csv(rna_data_path, sep="\t").dropna()
    rna_data = rna_data.drop(['Entrez_Gene_Id'], axis=1)

    # Extract common genes
    common_genes = set(cna_data['Hugo_Symbol']) & set(rna_data['Hugo_Symbol'])
    common_with_PAM50 = common_genes & set(PAM50_genes)
    common_genes = pandas.Series(list(common_genes)).dropna()
    cna_data = cna_data.loc[cna_data['Hugo_Symbol'].isin(common_genes)]
    rna_data = rna_data.loc[rna_data['Hugo_Symbol'].isin(common_genes)]

    # Extract common patients
    common_cols = cna_data.columns.intersection(rna_data.columns)
    cna_data = cna_data[common_cols]
    rna_data = rna_data[common_cols]

    # Sort by gene
    cna_data = cna_data.sort_values(by='Hugo_Symbol')
    rna_data = rna_data.sort_values(by='Hugo_Symbol')

    # Extract most high-varied genes
    np_gene_data = rna_data.iloc[:, 1:].values
    top_MAD_cna = np.argsort(np.apply_along_axis(func1d=MAD, axis=1, arr=np_gene_data))[-1200:]

    # For random selection:
    # np.random.shuffle(top_MAD_cna)
    # top_MAD_cna = top_MAD_cna[:1200]

    # Obtain list of genes to extract
    selected_genes = cna_data.iloc[top_MAD_cna, 0]
    selected_genes = list(set(selected_genes) | common_with_PAM50)
    selected_genes = pandas.Series(list(selected_genes)).dropna()
    rna_data = rna_data.loc[rna_data['Hugo_Symbol'].isin(selected_genes)]
    cna_data = cna_data.loc[cna_data['Hugo_Symbol'].isin(selected_genes)]

    np_gene_data = cna_data.iloc[:, 1:].values
    top_MAD_cna = np.argsort(np.apply_along_axis(func1d=entropy, axis=1, arr=np_gene_data))[-300:]

    # For random selection:
    # np.random.shuffle(top_MAD_cna)
    # top_MAD_cna = top_MAD_cna[:300]

    selected_genes = cna_data.iloc[top_MAD_cna, 0]
    cna_data = cna_data.loc[cna_data['Hugo_Symbol'].isin(selected_genes)]

    # Convert CNA to one-hot encoding
    cna_data = cna_data.iloc[:, 1:]
    cna_data = cna_data.applymap(normalize_cna)
    cna_data = cna_data.transpose()
    cna_data = pandas.get_dummies(cna_data, columns=cna_data.columns)
    cna_data = cna_data.transpose()

    # Remove gene column from RNA
    rna_data = rna_data.iloc[:, 1:]


    # Get number of features
    n_cna_features = cna_data.shape[0]
    n_rna_features = rna_data.shape[0]
    print("CNA features: ", n_cna_features)
    print("RNA features: ", n_rna_features)

    np_type_data = []
    np_rna_data = []
    np_cna_data = []


    for index, row in intclust_data.iterrows():

        patient_id = row['PATIENT_ID']
        cluster_id = row['INTCLUST']

        # Merge cluster 4
        if cluster_id == '4ER+' or cluster_id == '4ER-':
            cluster_id = 4

        # Exclude clusters 2 and 6
        if cluster_id == '2' or cluster_id == '6':
            continue

        cluster_id = int(cluster_id) - 1

        if patient_id in rna_data:

            # Check if number of elements per cluster is exceeded
            unique, counts = np.unique(np_type_data, return_counts=True)
            count_dict = dict(zip(unique, counts))
            if cluster_id in count_dict and count_dict[cluster_id] >= 200:
                continue

            rna_sample = rna_data[patient_id].values.transpose()
            cna_sample = cna_data[patient_id].values.transpose()

            np_rna_data.append(rna_sample)
            np_cna_data.append(cna_sample)
            np_type_data.append(cluster_id)

    np_rna_data = np.array(np_rna_data)
    np_cna_data = np.array(np_cna_data)
    np_type_data = np.array(np_type_data)


    # Normalize RNA data
    np_rna_data = 2 * (np_rna_data - np.min(np_rna_data)) / (np.max(np_rna_data) - np.min(np_rna_data)) - 1

    # Print cluster counts
    unique, counts = np.unique(np_type_data, return_counts=True)
    print(counts)


    # Split into training and test data
    n_samples = np_rna_data.shape[0]
    n_train_samples = int(n_samples * 0.8)
    sample_indices = np.arange(n_samples)
    np.random.shuffle(sample_indices)
    train_indices = sample_indices[:n_train_samples]
    test_indices = sample_indices[n_train_samples:]


    X_train_rna = np_rna_data[train_indices, :].copy()
    X_train_cna = np_cna_data[train_indices, :].copy()
    y_train = np_type_data[train_indices].copy()

    X_test_rna = np_rna_data[test_indices, :].copy()
    X_test_cna = np_cna_data[test_indices, :].copy()
    y_test = np_type_data[test_indices].copy()


    # For setting random RNA genes to zero:
    for i in range(X_test_rna.shape[0]):
        zero_indices = np.arange(1200)
        np.random.shuffle(zero_indices)
        zero_indices = zero_indices[:120]
        X_test_rna[i:i+1, zero_indices] = 0


# ----------------------------------------Multi-Modal AutoEncoder---------------------------------------------------

    def run_multi_encoder(n_multi_epochs, verb):

        # Define layers
        rna_hidden = 800
        input_rna = Input(shape=(n_rna_features,))
        hidden_rna_layer_1 = Dense(rna_hidden, activation='sigmoid')

        cna_hidden = 800
        input_cna = Input(shape=(n_cna_features,))
        hidden_cna_layer_1 = Dense(cna_hidden, activation='sigmoid')

        enc_features = 1600
        combined_layer = Dense(enc_features, activation='sigmoid')

        hidden_rna_layer_2 = Dense(rna_hidden, activation='sigmoid')
        output_rna_layer = Dense(n_rna_features, activation='sigmoid')

        hidden_cna_layer_2 = Dense(cna_hidden, activation='sigmoid')
        output_cna_layer = Dense(n_cna_features, activation='sigmoid')


        # Train first set of layers
        hidden_rna = hidden_rna_layer_1(input_rna)
        output_rna = output_rna_layer(hidden_rna)
        autoencoder = Model(input_rna, output_rna)
        autoencoder.compile(loss='mse', optimizer=optimizers.SGD(lr=0.01))
        autoencoder.fit(X_train_rna, X_train_rna,
                        epochs=n_multi_epochs, batch_size=32, shuffle=True, verbose=0)
        hidden_rna_layer_1.trainable = False
        rna_hidden_encoder = Model(input_rna, hidden_rna)
        intermediate_rna = rna_hidden_encoder.predict(X_train_rna)

        hidden_cna = hidden_cna_layer_1(input_cna)
        output_cna = output_cna_layer(hidden_cna)
        autoencoder = Model(input_cna, output_cna)
        autoencoder.compile(loss='mse', optimizer=optimizers.SGD(lr=0.01))
        autoencoder.fit(X_train_cna, X_train_cna,
                        epochs=n_multi_epochs, batch_size=32, shuffle=True, verbose=0)
        hidden_cna_layer_1.trainable = False
        cna_hidden_encoder = Model(input_cna, hidden_cna)
        intermediate_cna = cna_hidden_encoder.predict(X_train_cna)


        # Train combined layer
        hidden_rna = hidden_rna_layer_1(input_rna)
        hidden_cna = hidden_cna_layer_1(input_cna)
        concat = Concatenate()([hidden_rna, hidden_cna])
        combined = combined_layer(concat)
        output_rna = hidden_rna_layer_2(combined)
        output_cna = hidden_cna_layer_2(combined)
        autoencoder = Model([input_rna, input_cna], [output_rna, output_cna])
        autoencoder.compile(loss = ['mse', 'mse'] , optimizer=optimizers.SGD(lr=0.01))
        autoencoder.fit([X_train_rna, X_train_cna], [intermediate_rna, intermediate_cna],
                        epochs=n_multi_epochs, batch_size=32, shuffle=True, verbose=0)
        combined_layer.trainable = False



        # Train full model
        hidden_rna = hidden_rna_layer_1(input_rna)
        hidden_cna = hidden_cna_layer_1(input_cna)
        concat = Concatenate()([hidden_rna, hidden_cna])
        combined = combined_layer(concat)
        hidden_rna_2 = hidden_rna_layer_2(combined)
        hidden_cna_2 = hidden_cna_layer_2(combined)
        output_rna = output_rna_layer(hidden_rna_2)
        output_cna = output_cna_layer(hidden_cna_2)

        autoencoder = Model([input_rna, input_cna], [output_rna, output_cna])
        autoencoder.compile(loss=['mse', 'mse'], optimizer= optimizers.SGD(lr=0.01))

        autoencoder.fit([X_train_rna, X_train_cna], [X_train_rna, X_train_cna],
                        epochs=n_multi_epochs, batch_size=32, shuffle=True, verbose=0)

        multi_encoder = Model([input_rna, input_cna], combined)

        multi_enc_train = multi_encoder.predict([X_train_rna, X_train_cna])
        multi_enc_test  = multi_encoder.predict([X_test_rna, X_test_cna])


        # Evaluate different representations
        entry = []
        entry.append(run_complex_classifier(multi_enc_train, multi_enc_test))
        entry.append(run_complex_classifier(X_train_rna, X_test_rna))
        entry.append(run_complex_classifier(X_train_cna, X_test_cna))
        entry.append(run_complex_classifier(np.hstack((X_train_rna, X_train_cna)), np.hstack((X_test_rna, X_test_cna))))

        entry.append(run_simple_classifier(multi_enc_train, multi_enc_test))
        entry.append(run_simple_classifier(X_train_rna, X_test_rna))
        entry.append(run_simple_classifier(X_train_cna, X_test_cna))
        entry.append(run_simple_classifier(np.hstack((X_train_rna, X_train_cna)), np.hstack((X_test_rna, X_test_cna))))

        print(entry)

        data.append(entry)

        return True


#----------------------------------------Classifier-------------------------------------------------------

    def run_complex_classifier(x_train, x_test):

        classifier = GradientBoostingClassifier(n_estimators=100, max_features='log2', random_state=0).fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        return accuracy_score(y_test, y_pred)


    def run_simple_classifier(x_train, x_test):

        classifier = AdaBoostClassifier(n_estimators=100, random_state=0).fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        return accuracy_score(y_test, y_pred)

# ----------------------------------------Runner-------------------------------------------------------

    run_multi_encoder(verb=0, n_multi_epochs=200)




# Run for 15 iterations
for i in range(15):
    print("Iteration ", i, "...")
    train_graph()
    print("")
    print("")


# Obtain averages
data = np.array(data)

means, deviations = np.apply_along_axis(func1d=np.mean, axis=0, arr=data), \
                    np.apply_along_axis(func1d=np.std, axis=0, arr=data)

print(means)
print(deviations)
