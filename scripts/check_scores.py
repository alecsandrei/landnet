import pandas as pd
from landnet.enums import GeomorphometricalVariable
from landnet.config import MODELS_DIR, TRIAL_NAME,PROCESSED_DATA_DIR
import re

REGEX = re.compile(r'\d+\.\d+')

def get_value_from_tensor_string(tensor_as_string: str):
    return float(REGEX.search(tensor_as_string).group(0))


def check_infered_metrics():
    frames = []
    for csv in (MODELS_DIR / TRIAL_NAME).glob('*csv'):
        if not csv.parent.name == 'test':
            continue
        variable = GeomorphometricalVariable(csv.parents[2].name.split('_')[0])
        metrics = pd.read_csv(csv)
        metrics = metrics.set_axis(metrics.iloc[:, 0], axis='index').iloc[:, 1]
        metrics = metrics.apply(get_value_from_tensor_string)
        metrics.name = variable.value
        as_frame = metrics.to_frame().T
        print(as_frame)
        frames.append(as_frame)
    
    resulted_metrics = pd.concat(frames, ignore_index=False).sort_values('sensitivity', ascending=False)
    resulted_metrics.to_csv(PROCESSED_DATA_DIR / f'{TRIAL_NAME}_results.csv')


if __name__ == '__main__':
    frames = []
    sort_by = 'sensitivity'
    for csv in (MODELS_DIR / TRIAL_NAME).glob('*csv'):
        metrics = pd.read_csv(csv)
        variable = GeomorphometricalVariable(csv.name.split('_')[0])
        metrics.columns = metrics.columns.str.replace('val_', '')
        metrics = metrics[['f1_score', 'sensitivity', 'specificity', 'accuracy']].sort_values(sort_by, ascending=False)
        print(metrics.T)
    
    resulted_metrics = pd.concat(frames, ignore_index=False).sort_values('sensitivity', ascending=False)
    resulted_metrics.to_csv(PROCESSED_DATA_DIR / f'{TRIAL_NAME}_results.csv')
        
    