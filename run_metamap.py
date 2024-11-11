from pymetamap import MetaMap
import os
import time
import pandas as pd
import multiprocessing as mp
import argparse

# These semantic types are used to restrict the search to symptom-related concepts
# See https://metamap.nlm.nih.gov/Docs/SemanticTypes_2018AB.txt for a full list of semantic types
# Selected symptom concepts are those used in SympGraph: https://dl.acm.org/doi/abs/10.1145/2339530.2339712
# like in the original BD4H paper.
symptom_related_types = ["sosy", "dsyn", "neop", "fngs", "bact", "virs", "cgab", 
                        "acab", "lbtr", "inpo", "mobd", "comd", "anab"]

# Columns we want from Fielded MetaMap Indexing (MMI) Output
keys_of_interest = ['score', 'preferred_name', 'cui', 'semtypes', 'trigger', 'pos_info']
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
    parser.add_argument('--start_stop_servers', '-s', action='store_true', help=start_stop_desc)
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

def process_note(discharge_summary, patient_id):
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
    results_df = pd.DataFrame(cols, columns=keys_of_interest)

    for i, row in results_df.iterrows():
        # remove '[' and ']' from the trigger
        row['trigger'] = row['trigger'][1:-1] 

        # Remove negated concepts
        triggers = [trigger for trigger in row['trigger'].split(',')  if trigger[-1] == '0']

        # remove rows that only had negated concepts
        if not triggers:
            results_df.drop(i, inplace=True)

    # Add patientID column and drop trigger column
    results_df['patientID'] = patient_id
    results_df.drop(columns=['trigger'], inplace=True)
    results_df = results_df[['patientID', 'preferred_name']]

    return results_df

def update_progress(result, results, counter, total_tasks, i):
    """
    Progress updater
    """
    if not result.empty:
        results.append(result)
    counter.append(i)
    print(f'Progress: {len(counter)}/{total_tasks} notes processed.')

def main(args):
    """
    Main function for processing files with MetaMap.
    
    :param args: Command line arguments
    """
    # Start MetaMap servers (uncomment if you want to start the servers here)
    if args.start_stop_servers:
        start_mm_servers(mm_base_dir)

    # Read discharge summaries from CSV file
    dataset_df = pd.read_csv('data/summaries.csv')

    # Get discharge summaries and patient IDs
    example_texts = dataset_df['text'].tolist()
    patient_ids = dataset_df['patientID'].tolist()

    # Process discharge summaries using multiprocessing
    num_processes = mp.cpu_count()
    total_tasks = len(example_texts)
    results = [] # list of dataframes ('patientID', 'symptom')
    counter = [] # list of processed texts
    with mp.Pool(processes=num_processes) as pool:
        for i, (text, patient_id) in enumerate(zip(example_texts, patient_ids)):
            pool.apply_async(
                process_note, 
                args=(text, patient_id), 
                callback=lambda result: update_progress(result, results, counter, total_tasks, i)
            )
        pool.close()
        pool.join()

    # Concatenate results
    results_df = pd.concat(results, ignore_index=True).sort_values(by='patientID')
    # Save results 
    results_df.to_csv('data/symptoms.csv', index=False)

    # Stop MetaMap servers (uncomment if you want to stop the servers here)
    if args.start_stop_servers:
        stop_mm_servers(mm_base_dir)

if __name__ == '__main__':
    args = get_args()
    main(args)
