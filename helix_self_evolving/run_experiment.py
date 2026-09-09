from src.data_loader import load_allen_data, preprocess
from src.models import StaticModel, NaiveStreamingModel, HELIXModel
import numpy as np
import pandas as pd

def run():
    cells_df, adata = load_allen_data()
    X, y, stages = preprocess(cells_df, adata)

    stage_names = sorted(set(stages))[:5]   # take first five
    stage_data = {s: (X[stages==s], y[stages==s]) for s in stage_names}

    results = {'static': [], 'naive': [], 'helix': []}
    for approach in ['static', 'naive', 'helix']:
        if approach == 'static':
            model = StaticModel()
            X0, y0 = stage_data[stage_names[0]]
            model.fit(X0, y0)
            for s in stage_names:
                X_test, y_test = stage_data[s]
                acc = np.mean(model.predict(X_test) == y_test)
                results['static'].append(acc)
        elif approach == 'naive':
            model = NaiveStreamingModel()
            for s in stage_names:
                X_train, y_train = stage_data[s]
                model.partial_fit(X_train, y_train) 
                acc = np.mean(model.predict(X_train) == y_train)
                results['naive'].append(acc)
        else:  
            model = HELIXModel()
            for s in stage_names:
                X_train, y_train = stage_data[s]
                model.process_batch(X_train, y_train)
                acc = np.mean(model.model.predict(X_train) == y_train)
                results['helix'].append(acc)

    df = pd.DataFrame(results, index=stage_names)
    df.to_csv('results.csv')
    print(df)

if __name__ == '__main__':
    run()
