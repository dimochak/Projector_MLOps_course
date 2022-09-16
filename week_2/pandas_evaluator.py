import os.path
import pandas as pd
import timeit
import plotly.graph_objects as go

def run_load_experiment(path_to_file: str):
    start = timeit.default_timer()
    df = pd.read_csv(path_to_file)
    stop = timeit.default_timer()
    return round(stop - start, 5), df


def run_save_experiment(df: pd.DataFrame, path_to_save: str):
    start = timeit.default_timer()
    df.to_csv(path_to_save)
    stop = timeit.default_timer()
    return round(stop - start, 5)


def make_plots(trace_params: dict, fig_params: dict, path_to_reports: str):
    fig = go.Figure()
    num_of_traces = len(trace_params.keys())

    for i in range(num_of_traces):
        plot_params = trace_params[i]
        fig.add_trace(
            go.Scatter(
                x=plot_params['x_vals'],
                y=plot_params['y_vals'],
                mode=plot_params['mode'],
                name=plot_params['name']
            )
        )

    fig.update_layout(
        title=fig_params['title'],
        xaxis_title=fig_params['xaxis_title'],
        yaxis_title=fig_params['yaxis_title']
    )

    fig.write_html(path_to_reports)


if __name__ == '__main__':
    lines_in_file = [10000, 100000, 500000, 1000000]
    file_prefix = 'test_data_'
    reading_results = []
    saving_results = []
    for number_of_lines in lines_in_file:
        path_to_file = os.path.join(os.getcwd(), 'pr_3', 'test_data', f'{file_prefix}{number_of_lines}.csv')
        res, df = run_load_experiment(path_to_file)
        reading_results.append(res)
        print(f'Loading file with {number_of_lines} number of lines took {res} seconds')
        res = run_save_experiment(df, path_to_file)
        print(f'Saving file with {number_of_lines} number of lines took {res} seconds')
        saving_results.append(res)

    trace_dict = {
        0: {
            'x_vals': lines_in_file,
            'y_vals': reading_results,
            'mode': 'lines+markers',
            'name': 'Loading results'
        },
        1: {
            'x_vals': lines_in_file,
            'y_vals': saving_results,
            'mode': 'lines+markers',
            'name': 'Saving results'
        }
    }

    fig_dict = {
        'title': 'Comparing results of loading and saving csv via Pandas',
        'xaxis_title': 'Num of lines in file',
        'yaxis_title': 'Seconds'
    }

    make_plots(trace_dict, fig_dict, os.path.join(os.getcwd(), 'pr_3', 'reports', 'benchmarking.html'))

