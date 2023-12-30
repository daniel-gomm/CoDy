import pandas as pd
import sys
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Format UCI Data')
    
    parser.add_argument('--input', required=True, type=str, help='Path to the input file')
    parser.add_argument('--output', required=True, type=str, help='Path to which the output is saved to')
    
    try:
        args = parser.parse_args()
    except SystemExit:
        parser.print_help()
        sys.exit(0)
    
    raw_data = pd.read_csv(args.input, sep=" ", header=None)
    
    raw_data.columns = ['timestamp', 'item_id', 'user_id', 'state_label']
    raw_data['timestamp'] = pd.to_datetime(raw_data['timestamp'])
    raw_data['timestamp'] = (raw_data['timestamp'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    reordered_data = raw_data[['user_id', 'item_id', 'timestamp', 'state_label']]
    reordered_data.to_csv(args.output, index=False)

    print(f'Reformatted file saved to {args.output}')

