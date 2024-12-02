import pandas as pd
from collections import defaultdict

def count_symptoms(df):
    """
    Count the number of times each symptom appears in the dataset.

    :param df: The DataFrame containing the symptoms.

    :return: A dictionary mapping each symptom to the number of times it appears.
    """
    symptom_counts = defaultdict(int)
    for symptoms in df['preferred_name']:
        for symptom in symptoms:
            symptom_counts[symptom] += 1
    return symptom_counts

def filter_symptoms(symptoms, to_delete):
    """
    Filter out symptoms that appear less than 10 times in the dataset.

    :param symptoms: The list of symptoms to filter.
    :param to_delete: The set of symptoms to delete.

    :return: The filtered list of symptoms.
    """
    return [symptom for symptom in symptoms if symptom not in to_delete]

def filter_less_frequent(filename):
    """
    Combine symptoms for rows with the same SUBJECT_ID and HADM_ID, and
    filter out symptoms that appear less than 10 times in the dataset.

    :param filename: The filename of the CSV file containing the symptoms.

    :return: None
    """
    # SUBJECT_ID,HADM_ID,preferred_name
    # int, int, str of list
    df = pd.read_csv(filename)

    df['preferred_name'] = df['preferred_name'].apply(lambda x: eval(x) if isinstance(x, str) else [])

    # Combine lists for rows with the same SUBJECT_ID and HADM_ID
    df = (
        df.groupby(['SUBJECT_ID', 'HADM_ID'], as_index=False)
        .agg({'preferred_name': lambda x: sum(x, [])})  # Combine lists
    )

    # Count the number of times each symptom appears in the dataset
    symptom_counts = count_symptoms(df)
    print(f"Found {len(symptom_counts)} unique symptoms")

    # Identify symptoms to delete (any appearing less than 10 times)
    print('Identifying symptoms to delete')
    symptoms_to_delete = {symptom for symptom, count in symptom_counts.items() if count < 10}
    print(f"Deleting {len(symptoms_to_delete)} symptoms")

    # Filter out the symptoms to delete
    df['preferred_name'] = df['preferred_name'].apply(lambda x: filter_symptoms(x, symptoms_to_delete))

    # Reformat the list to a string representation with single quotes
    df['preferred_name'] = df['preferred_name'].apply(lambda x: str(x))

    # Go through each row and if the preferred_name is empty, remove the row
    print('Removing rows with empty preferred_name')
    df = df[df['preferred_name'] != '[]']

    # Save the modified DataFrame to a new CSV file
    df.to_csv(filename, index=False)

if __name__ == '__main__':
    filter_less_frequent('mimic_data/symptoms_top_50.csv')