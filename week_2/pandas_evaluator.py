import os.path
import pandas as pd
import timeit
import plotly.graph_objects as go


def run_load_experiment(path_to_file: str, file_type: str):
    if file_type == 'csv':
        start = timeit.default_timer()
        df = pd.read_csv(path_to_file)
        stop = timeit.default_timer()
    elif file_type == 'h5':
        start = timeit.default_timer()
        df = pd.read_hdf(path_to_file, key='data')
        stop = timeit.default_timer()
    else:
        start = timeit.default_timer()
        df = pd.read_parquet(path_to_file)
        stop = timeit.default_timer()
    return round(stop - start, 5), df


def run_save_experiment(df: pd.DataFrame, path_to_save: str, file_type: str):
    if file_type == 'csv':
        start = timeit.default_timer()
        df.to_csv(path_to_save)
        stop = timeit.default_timer()
    elif file_type == 'h5':
        start = timeit.default_timer()
        df.to_hdf(path_to_save, key='data')
        stop = timeit.default_timer()
    else:
        start = timeit.default_timer()
        df.to_parquet(path_to_save)
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
    file_formats = ['csv', 'h5', 'parquet']
    file_prefix = 'test_data'
    reading_results = []
    saving_results = []
    trace_dict = {}
    unique_ids = iter(range(len(lines_in_file)*len(file_formats)))
    for file_format in file_formats:
        for number_of_lines in lines_in_file:
            path_to_file = os.path.join(os.getcwd(), 'pr_3', 'test_data', f'{file_prefix}_{number_of_lines}.{file_format}')
            res, df = run_load_experiment(path_to_file, file_format)
            reading_results.append(res)
            print(f'Loading .{file_format} file with {number_of_lines} number of lines took {res} seconds')
            res = run_save_experiment(df, path_to_file, file_format)
            print(f'Saving .{file_format} file with {number_of_lines} number of lines took {res} seconds')
            saving_results.append(res)

        trace_dict[next(unique_ids)] = {
            'x_vals': lines_in_file,
            'y_vals': reading_results,
            'mode': 'lines+markers',
            'name': f'Loading results (.{file_format} format)'
        }
        trace_dict[next(unique_ids)] = {
            'x_vals': lines_in_file,
            'y_vals': saving_results,
            'mode': 'lines+markers',
            'name': f'Saving results (.{file_format} format)'
        }
        reading_results = []
        saving_results = []

    fig_dict = {
        'title': 'Comparing results of loading and saving different file formats',
        'xaxis_title': 'Num of lines in file',
        'yaxis_title': 'Seconds'
    }

    make_plots(trace_dict, fig_dict, os.path.join(os.getcwd(), 'pr_3', 'reports', 'benchmarking.html'))
