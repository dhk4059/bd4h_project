from pymetamap import MetaMap
import os
import time
import pandas as pd
import multiprocessing as mp
import argparse
import matplotlib.pyplot as plt

# These semantic types are used to restrict the search to symptom-related concepts
# See https://metamap.nlm.nih.gov/Docs/SemanticTypes_2018AB.txt for a full list of semantic types
# Selected symptom concepts are those used in SympGraph: https://dl.acm.org/doi/abs/10.1145/2339530.2339712
# like in the original BD4H paper.
symptom_related_types = ["sosy", "dsyn", "neop", "fngs", "bact", "virs", "cgab", 
                        "acab", "lbtr", "inpo", "mobd", "comd", "anab"]

# Columns we want from Fielded MetaMap Indexing (MMI) Output
# keys_of_interest = ['score', 'preferred_name', 'cui', 'semtypes', 'trigger', 'pos_info']
keys_of_interest = ['preferred_name', 'trigger']

# Metamap PATH variables
mm_base_dir = '/home/qwer/metamap/public_mm/' # change this to your MetaMap installation directory
mm_exe = 'bin/metamap20'
mm_pos_server_exe = 'bin/skrmedpostctl'
mm_wsd_server_exe = 'bin/wsdserverctl'

def get_args():
    """
    Get command line arguments
    """
    main_desc = """Process discharge summaries using MetaMap.
This script processes discharge summaries using MetaMap 
and extracts symptom-related concepts from summaries.csv file."""
    parser = argparse.ArgumentParser(description=main_desc, formatter_class=argparse.RawTextHelpFormatter)
    start_stop_desc = """Including this flag will start and stop 
MetaMap servers inside this script."""
    parser.add_argument('-s', '--start_stop_servers', action='store_true', help=start_stop_desc, default=False)
    parser.add_argument('-n', '--top_N', type=int, help='Number of top ICD9 codes to consider', default=50)
    return parser.parse_args()

def start_mm_servers(mm_base_dir):
    """
    Start MetaMap servers and wait for them to start
    """
    print('Starting the servers...')
    print('=' * 80)
    os.system(mm_base_dir + mm_pos_server_exe + ' start')
    os.system(mm_base_dir + mm_wsd_server_exe + ' start')
    print('=' * 80)
    print()

    wait_time = 60
    print(f'Waiting {wait_time} sec for the servers to start...')
    time.sleep(wait_time)

def stop_mm_servers(mm_base_dir):
    """
    Stop MetaMap servers
    """
    print('Stopping the servers...')
    print('=' * 80)
    os.system(mm_base_dir + mm_pos_server_exe + ' stop')
    os.system(mm_base_dir + mm_wsd_server_exe + ' stop')
    print('=' * 80)
    print()

def get_keys_from_mm(concept, klist):
    """
    Get keys of interest from a MetaMap Fielded MetaMap Indexing (MMI) Output concept

    :param concept: A ConceptMMI from MetaMap Fielded MetaMap Indexing (MMI) Output
    :param klist: A list of keys to extract from the concept

    :return: A tuple of values extracted from the concept
    """
    conc_dict = concept._asdict()
    conc_list = [conc_dict.get(kk) for kk in klist]
    return tuple(conc_list)

def process_note(discharge_summary, patient_id, hadm_id):
    """
    Process a discharge summary using MetaMap and extract symptom-related concepts

    :param discharge_summary: A discharge summary text

    :return: A DataFrame containing symptom-related concepts extracted from the discharge summary
    """
    # Init MetaMap instance
    mm = MetaMap.get_instance(mm_base_dir + mm_exe)

    cons, errs = mm.extract_concepts(
        [discharge_summary],
        word_sense_disambiguation=True,   # -y
        allow_large_n=True,               # -l
        ignore_word_order=True,           # -i
        restrict_to_sts=symptom_related_types, # -J
        prune=30                          # recommended pruning value for memory efficiency.
    )

    # Extract keys of interest from the concepts
    cols = [get_keys_from_mm(cc, keys_of_interest) for cc in cons]
    results_df = pd.DataFrame(cols, columns=keys_of_interest).head(50)
    if len(results_df) == 1:
        print(f'Only 1 symptom found for patient {patient_id}. Skipping...')
        return pd.DataFrame()

    for i, row in results_df.iterrows():
        # remove '[' and ']' from the trigger
        # if pd.isnull(row['trigger']):
            # return pd.DataFrame()
        row['trigger'] = row['trigger'][1:-1] 

        # Remove negated concepts
        triggers = [trigger for trigger in row['trigger'].split(',')  if trigger[-1] == '0']

        # remove rows that only had negated concepts
        if not triggers:
            results_df.drop(i, inplace=True)

    # Add SUBJECT_ID and HADM_ID columns
    results_df['SUBJECT_ID'] = patient_id
    results_df['HADM_ID'] = hadm_id

    # Drop the trigger column
    results_df.drop(columns=['trigger'], inplace=True)
    results_df = results_df[['SUBJECT_ID', 'HADM_ID', 'preferred_name']]
    print(f'Patient ID: {patient_id}, HADM ID: {hadm_id}, Number of symptoms: {len(results_df)}')

    results_df = results_df.groupby(["SUBJECT_ID", "HADM_ID"])['preferred_name'].apply(list).reset_index()
    # results_df['HADM_ID'] = results_df['HADM_ID'].astype(int)

    return results_df

def update_progress(result, results, counter, total_tasks, i):
    """
    Progress updater
    """
    if not result.empty:
        results.append(result)
    counter.append(i)
    print(f'Progress: {len(counter)}/{total_tasks} notes processed.')

def run_metamap(start_stop_servers, dataset_df, top_N=50):
    """
    Main function for processing files with MetaMap.
    
    :param start_stop_servers (bool): Flag to start and stop MetaMap servers
    :param dataset_df (pd.DataFrame): DataFrame with 'SUBJECT_ID' and 'TEXT' columns, preprocessed
                                      from NOTEEVENTS table in MIMIC-III dataset.
    :param top_N (int): Number of top most-common diseases to consider

    :return: none
    """
    # Start MetaMap servers (uncomment if you want to start the servers here)
    if start_stop_servers:
        start_mm_servers(mm_base_dir)

    # Get discharge summaries and patient IDs
    summary_texts = dataset_df['TEXT'].tolist()
    patient_ids = dataset_df['SUBJECT_ID'].tolist()
    hadm_ids = dataset_df['HADM_ID'].tolist()

    # Process discharge summaries using multiprocessing
    num_processes = mp.cpu_count()
    total_tasks = len(summary_texts)
    results = [] # list of dataframes ('SUBJECT_ID', 'symptom')
    counter = [] # list of processed texts
    with mp.Pool(processes=num_processes) as pool:
        for i, (text, patient_id, hadm_id) in enumerate(zip(summary_texts, patient_ids, hadm_ids)):
            pool.apply_async(
                process_note, 
                args=(text, patient_id, hadm_id),
                callback=lambda result: update_progress(result, results, counter, total_tasks, i)
            )
        pool.close()
        pool.join()

    # Concatenate results
    results_df = pd.concat(results, ignore_index=True)

    # Sort by both SUBJECT_ID and HADM_ID
    results_df.sort_values(by=['SUBJECT_ID', 'HADM_ID'], inplace=True)

    # Save results 
    results_df.to_csv(f'mimic_data/symptoms_top_{top_N}.csv', index=False)

    # Stop MetaMap servers (uncomment if you want to stop the servers here)
    if start_stop_servers:
        stop_mm_servers(mm_base_dir)

def preprocess_noteevents(top_N=50):
    """
    Preprocess the NOTEEVENTS.csv file to filter only patient IDs and discharge summaries

    :param top_N (int): Number of top most-common diseases to consider

    :return: A DataFrame containing 'SUBJECT_ID', 'HADM_ID', and 'TEXT' columns
    """
    print('Preprocessing NOTEEVENTS.csv...')

    # Read the DIAGNOSES_ICD table with only relevant columns
    cols_to_read = ['SUBJECT_ID', 'SEQ_NUM', 'ICD9_CODE']
    cols_to_read = ['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE']
    print('Reading DIAGNOSES_ICD.csv...')
    diag_icd_df = pd.read_csv('mimic_data/DIAGNOSES_ICD.csv', usecols=cols_to_read, dtype={'ICD9_CODE': str})

    # Drop any rows with empty cols
    diag_icd_df.dropna(subset=['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE'], inplace=True)

    # Drop columns where ICD9_CODE starts with 'E' or 'V', since these are not diseases
    diag_icd_df['ICD9_CODE'] = diag_icd_df['ICD9_CODE'].astype(str)
    diag_icd_df = diag_icd_df[~diag_icd_df['ICD9_CODE'].str.startswith(('E', 'V'))]

    # For each ICD9_CODE, replace with the first 3 characters of the code to keep disease category
    diag_icd_df['ICD9_CODE'] = diag_icd_df['ICD9_CODE'].apply(lambda x: str(x)[:3])

    # get the N-most common icd9 codes
    icd9_common_N = diag_icd_df['ICD9_CODE'].value_counts().head(top_N)

    # get only the rows with the 50-most common icd9 codes
    diag_icd_df = diag_icd_df[diag_icd_df['ICD9_CODE'].isin(icd9_common_N.index)]

    print(f'Filtered diagnosis codes for the top {top_N} most common ICD9 codes.')
    print()
    
    st = time.time()
    # Read the NOTEEVENTS table with only relevant columns
    cols_to_read = ['SUBJECT_ID', 'HADM_ID', 'CATEGORY', 'DESCRIPTION', 'TEXT']
    print('Reading NOTEEVENTS.csv...')
    noteevents_df = pd.read_csv('mimic_data/NOTEEVENTS.csv', usecols=cols_to_read)

    # drop any rows with empty cols
    noteevents_df.dropna(subset=['SUBJECT_ID', 'HADM_ID', 'TEXT'], inplace=True)

    noteevents_df['SUBJECT_ID'] = noteevents_df['SUBJECT_ID'].astype(int)
    noteevents_df['HADM_ID'] = noteevents_df['HADM_ID'].astype(int)

    et = round(time.time() - st, 2)
    print(f"Time to read csv: {et} sec")
    st = time.time()

    # filter only rows where CATEGORY is 'Discharge summary'
    noteevents_df = noteevents_df[noteevents_df['CATEGORY'] == 'Discharge summary']
    # filter only rows where DESCRIPTION is 'Report'
    noteevents_df = noteevents_df[noteevents_df['DESCRIPTION'] == 'Report']

    # drop the CATEGORY and DESCRIPTION columns
    noteevents_df.drop(columns=['CATEGORY', 'DESCRIPTION'], inplace=True)

    # drop any rows with SUBJECT_ID that are not in diag_icd_df
    noteevents_df = noteevents_df[noteevents_df['SUBJECT_ID'].isin(diag_icd_df['SUBJECT_ID'])]

    # drop any rows with HADM_ID that are not in diag_icd_df
    noteevents_df = noteevents_df[noteevents_df['HADM_ID'].isin(diag_icd_df['HADM_ID'])]

    noteevents_df = noteevents_df.sort_values(by='TEXT', key=lambda x: x.str.len(), ascending=True)
    # reset the index
    noteevents_df.reset_index(drop=True, inplace=True)

    # NOTE: Due to proecssing time constraints, we will only process up to summary
    # text length of 10,000 characters. This is enough to get about 60% of the data.
    breakpoint = 10000
    breakpoint_idx = noteevents_df[noteevents_df["TEXT"].str.len() > breakpoint].index[0]
    print(f'First row index with text length greater than {breakpoint}: {breakpoint_idx}')

    et = round(time.time() - st, 2)
    print(f"Time to preprocess: {et} sec")

    print(f'Preprocessing complete. Length of NOTEEVENTS.csv after filtering: {len(noteevents_df)}')
    print()

    # Plot histogram of text lengths 
    text_lengths = noteevents_df['TEXT'].str.len()
    print(f'Minimum text length: {text_lengths.min()}')
    print(f'Maximum text length: {text_lengths.max()}')
    plt.figure(figsize=(10, 6))
    plt.hist(text_lengths, bins=100, color='skyblue', edgecolor='black')
    plt.title('Histogram of Discharge Summary Text Lengths')
    plt.xlabel('Text Length')
    plt.ylabel('Frequency')
    plt.savefig(f'plots/discharge_summary_text_lengths_top_{top_N}.png')
    plt.close()
    print('Plotted histogram')
    print()
    print()

    return noteevents_df.head(breakpoint_idx)

if __name__ == '__main__':
    # Create directories if they don't exist
    if not os.path.exists('mimic_data'):
        os.makedirs('mimic_data')

    # Exit if NOTEEVENTS.csv is not found
    if not os.path.exists('mimic_data/NOTEEVENTS.csv'):
        print("NOTEEVENTS.csv not found. Please download the MIMIC-III dataset and extract the CSV file in the 'mimic_data' folder.")
        exit(0)

    # get command line arguments
    args = get_args()                   

    # preprocess NOTEEVENTS.csv into smaller DataFrame 
    dataset_df = preprocess_noteevents(top_N=args.top_N)

    # Testing with a smaller dataset
    # dataset_df = dataset_df.head(50)
    # dataset_df = dataset_df.iloc[45000:45020]
    # dataset_df = dataset_df.tail(30)

    st = time.time()
    # run MetaMap on the DataFrame
    run_metamap(args.start_stop_servers, dataset_df, top_N=args.top_N)

    et = round(time.time() - st, 2)
    print()
    print(f"Time to run metamap: {et} sec")
