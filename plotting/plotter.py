from collections import OrderedDict
import math

import pandas as pd
import numpy as np
import scipy.stats
import scipy.interpolate
import statsmodels.api as sm
import statsmodels.stats.proportion

import plotly.express as px
import plotly.graph_objects as go
import matplotlib
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.pyplot


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

x_plotting_resolution = 200

grid_linewidth = 1.5
main_linewidth = 2
label_fontsize = 30
tick_fontsize = 20
defaultmarkersize = 80
legend_fontsize = 25


def add_plotting_data(df, columns):
    for column in set(columns):
        df[column + '_ci'] = df[[column, column+'_dataset_size']].apply(get_ci, axis=1)
    return df


def get_ci(df_row):
    acc = df_row[[x for x in df_row.axes[0] if '_dataset_size' not in x][0]]
    dataset_size = df_row[[x for x in df_row.axes[0] if '_dataset_size' in x][0]]
    acc = acc / 100
    lo, hi = clopper_pearson(acc * dataset_size, dataset_size)
    low, high = acc - lo, hi - acc
    low, high = low * 100, high * 100
    return (low, high)


def clopper_pearson(k, n, alpha=0.005):
    lo = scipy.stats.beta.ppf(alpha/2, k, n-k+1)
    hi = scipy.stats.beta.ppf(1 - alpha/2, k+1, n-k)
    return lo, hi


def run_bootstrap_linreg(xs, ys, num_bootstrap_samples, x_eval_grid, seed):
    rng = np.random.RandomState(seed)
    num_samples = xs.shape[0]
    result_coeffs = []
    result_y_grid_vals = []
    x_eval_grid_padded = np.stack([np.ones(x_eval_grid.shape[0]), x_eval_grid], axis=1)
    for ii in range(num_bootstrap_samples):
        cur_indices = rng.choice(num_samples, num_samples)
        cur_x = np.stack([np.ones(num_samples), xs[cur_indices]], axis=1)
        cur_y = ys[cur_indices]
        cur_coeffs = np.linalg.lstsq(cur_x, cur_y, rcond=None)[0]
        result_coeffs.append(cur_coeffs)
        cur_y_grid_vals = np.dot(x_eval_grid_padded, cur_coeffs)
        result_y_grid_vals.append(cur_y_grid_vals)
    return np.vstack(result_coeffs), np.vstack(result_y_grid_vals)


def get_bootstrap_cis(xs, ys, num_bootstrap_samples, x_eval_grid, seed, significance_level_coeffs=95, significance_level_grid=95):
    coeffs, y_grid_vals = run_bootstrap_linreg(xs, ys, num_bootstrap_samples, x_eval_grid, seed)
    result_coeffs = []
    result_grid_lower = []
    result_grid_upper = []
    percentile_lower_coeffs = (100.0 - significance_level_coeffs) / 2
    percentile_upper_coeffs = 100.0 - percentile_lower_coeffs
    percentile_lower_grid = (100.0 - significance_level_grid) / 2
    percentile_upper_grid = 100.0 - percentile_lower_grid
    for ii in range(coeffs.shape[1]):
        cur_lower = np.percentile(coeffs[:, ii], percentile_lower_coeffs, interpolation='lower')
        cur_upper = np.percentile(coeffs[:, ii], percentile_upper_coeffs, interpolation='higher')
        result_coeffs.append((cur_lower, cur_upper))
    for ii in range(x_eval_grid.shape[0]):
        cur_lower = np.percentile(y_grid_vals[:, ii], percentile_lower_grid, interpolation='lower')
        cur_upper = np.percentile(y_grid_vals[:, ii], percentile_upper_grid, interpolation='higher')
        result_grid_lower.append(cur_lower)
        result_grid_upper.append(cur_upper)
    return result_coeffs, result_grid_lower, result_grid_upper


def transform_acc(acc, transform='linear'):
    if type(acc) is list:
        acc = np.array(acc)
    if transform == 'linear':
        return acc
    elif transform == 'probit':
        return scipy.stats.norm.ppf(acc / 100.0)
    elif transform == 'logit':
        return np.log(np.divide(acc / 100.0, 1.0 - acc / 100.0))


def inv_transform_acc(acc, transform='linear'):
    if type(acc) is list:
        acc = np.array(acc)
    if transform == 'linear':
        return acc
    elif transform == 'probit':
        return scipy.stats.norm.cdf(acc) * 100.0
    elif transform == 'logit':
        return (np.exp(acc)/(1 + np.exp(acc)))*100 


def tick_locs(low, hi, step):
    res = []
    assert step > 0
    cur = -100
    while cur <= hi:
        if cur >= low:
            res.append(cur)
        cur += step
    return res


def model_scatter_plot(df, x_axis, y_axis, xlim, ylim, model_types, num_bootstrap_samples,
                       title, transform='logit', x_unit='top-1, %', y_unit='top-1, %', include_legend=True,
                       tick_multiplier=10, extra_x_ticks=[], extra_y_ticks=[], set_aspect=False,
                       return_ordered_names=False, x_label=None, y_label=None, figsize=(10, 10), 
                       include_ideal=True, fit_color="red", bl_value=None, 
                       bl_name=None, return_separate_legend=False, num_legend_columns=3, error_line_width=0.5, error_alpha=0.5,
                       alpha=0.5, x_tick_multiplier=None, y_tick_multiplier=None):

    assert (df[df.show_in_plot == True][x_axis] <= xlim[1]).all()
    assert (df[df.show_in_plot == True][x_axis] >= xlim[0]).all()
    assert (df[df.show_in_plot == True][y_axis] <= ylim[1]).all()
    assert (df[df.show_in_plot == True][y_axis] >= ylim[0]).all()
    print('ylim range / xlim range aspect ratio: ', (ylim[1] - ylim[0]) / (xlim[1] - xlim[0]))

    fig, ax = matplotlib.pyplot.subplots(1, figsize=figsize)
    ax.grid(which='both', color='lightgray', linestyle='-', linewidth=grid_linewidth)

    x_tick_multiplier = x_tick_multiplier if x_tick_multiplier is not None else tick_multiplier
    y_tick_multiplier = y_tick_multiplier if y_tick_multiplier is not None else tick_multiplier
    tick_loc_x = np.array(sorted(tick_locs(xlim[0], xlim[1], x_tick_multiplier) + extra_x_ticks))
    ax.set_xticks(transform_acc(tick_loc_x, transform))
    ax.set_xticklabels([str(int(loc)) for loc in tick_loc_x])
    tick_loc_y = np.array(sorted(tick_locs(ylim[0], ylim[1], y_tick_multiplier) + extra_y_ticks))
    ax.set_yticks(transform_acc(tick_loc_y, transform))
    ax.set_yticklabels([str(int(loc)) for loc in tick_loc_y])
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    df_line = df[df.use_for_line_fit == True]
    x_acc_line_trans = transform_acc(df_line[x_axis], transform)
    y_acc_line_trans = transform_acc(df_line[y_axis], transform)

    xs = np.linspace(transform_acc(xlim[0], transform), transform_acc(xlim[1], transform), x_plotting_resolution)
    lin_fit = scipy.stats.linregress(x_acc_line_trans, y_acc_line_trans)
    slope = lin_fit[0]
    intercept = lin_fit[1]
    lin_fit_ys = xs * slope + intercept
    idx = np.argmax(y_acc_line_trans)
    x_max = x_acc_line_trans[idx]
    y_max = y_acc_line_trans[idx]
    print("ymax",y_max)
    random_slope = y_max/x_max

    # y_pred = x_acc_line_trans * slope + intercept
    # from sklearn.metrics import r2_score
    # score = r2_score(y_acc_line_trans, y_pred)
    # print(f'R2 SCORE FOR {x_axis} vs {y_axis} is:', score)

    print(f'Slope {slope}, intercept {intercept}, r {lin_fit[2]}, pvalue {lin_fit[3]}, stderr {lin_fit[4]}')
    coeffs_ci, fit_lower, fit_upper = get_bootstrap_cis(x_acc_line_trans, y_acc_line_trans, num_bootstrap_samples, xs, 720257663,
                                                        significance_level_coeffs=95, significance_level_grid=95)
    print(f'Bootstrap CIs: {coeffs_ci}')
    sm_model = sm.OLS(y_acc_line_trans, np.stack([np.ones(x_acc_line_trans.shape[0]), x_acc_line_trans], axis=1))
    sm_results = sm_model.fit()
    print(sm_results.summary())

    ax.set_xlim([transform_acc(xlim[0], transform), transform_acc(xlim[1], transform)])
    ax.set_ylim([transform_acc(ylim[0], transform), transform_acc(ylim[1], transform)])
    if include_ideal:
        ideal_repro_line = ax.plot(xs, xs, linestyle='dashed', color='black', linewidth=main_linewidth, label='y = x')
    ax.fill_between(xs, fit_upper, fit_lower, color=f'tab:{fit_color}', alpha=0.3, zorder=6, edgecolor='none', linewidth=0.1)

    df_plot = df[df.show_in_plot == True]
    x_acc_plot = df_plot[x_axis]
    y_acc_plot = df_plot[y_axis]
    x_acc_ci = pd.DataFrame(df_plot[f'{x_axis}_ci'].tolist(), index=df_plot.index).to_numpy()
    y_acc_ci = pd.DataFrame(df_plot[f'{y_axis}_ci'].tolist(), index=df_plot.index).to_numpy()
    x_acc_plot_trans = transform_acc(x_acc_plot, transform)
    y_acc_plot_trans = transform_acc(y_acc_plot, transform)
    labels = df_plot.model_type.to_numpy()

    xerr_low = x_acc_plot_trans - transform_acc(x_acc_plot - x_acc_ci[:, 0], transform)
    xerr_high = transform_acc(x_acc_plot + x_acc_ci[:, 1], transform) - x_acc_plot_trans
    xerr = np.stack((xerr_low, xerr_high), axis=0)
    yerr_low = y_acc_plot_trans - transform_acc(y_acc_plot - y_acc_ci[:, 0], transform)
    yerr_high = transform_acc(y_acc_plot + y_acc_ci[:, 1], transform) - y_acc_plot_trans
    yerr = np.stack((yerr_low, yerr_high), axis=0)

    model_points = OrderedDict()
    model_names = []
    for m in model_types:
        if len(m.value) > 2:
            n, c, s = m.value
            if isinstance(s, tuple):
                s, a = s
            else:
                a = alpha
        else:
            n, c = m.value
            s = defaultmarkersize
            a = alpha
        if not any(labels == m):
            continue
        ax.errorbar(x_acc_plot_trans[labels == m], y_acc_plot_trans[labels == m], xerr=xerr[:, labels == m], yerr=yerr[:, labels == m],
                    capsize=2, linewidth=error_line_width, ls='none', color=c, alpha=error_alpha, zorder=8)
        # alpha = None if isinstance(c, (list, tuple)) and len(c) == 4 else 0.5
        points = ax.scatter(x_acc_plot_trans[labels == m], y_acc_plot_trans[labels == m],
                            zorder=9, color=c, s=s, label=n, alpha=a, linewidths=0)
        model_points[n] = points
        # Reverse models within groups, then reverse all models. This is the
        # path that gives us the order that matches matplotlib's artists.
        model_names.extend(list(reversed(df_plot[labels == m].index)))
    model_names = model_names[::-1]

    fit_line = ax.plot(xs, lin_fit_ys, color=f'tab:{fit_color}', zorder=7, linewidth=main_linewidth, label='Linear fit', alpha=0.5)
    ax.set_xlabel(f'{x_label if x_label else x_axis} ({x_unit})' , fontsize=label_fontsize)
    ax.set_ylabel(f'{y_label if y_label else y_axis} ({y_unit})', fontsize=label_fontsize)
    ax.set_title(f'{title}', fontsize=label_fontsize)
    if set_aspect:
        ax.set_aspect('equal', adjustable='box')
    if include_legend:
        if not include_ideal:
            ax.legend(list(model_points.values()) + [fit_line[0]],
                    list(model_points.keys()) + ['Linear fit'],
                    fontsize=label_fontsize, markerscale=1.5, frameon=False)
        else:
            ax.legend([ideal_repro_line[0]] + list(model_points.values()) + [fit_line[0]],
                    ['y = x'] + list(model_points.keys()) + ['Linear fit'],
                    fontsize=label_fontsize, markerscale=1.5, frameon=False)
    fig.tight_layout()

    if return_separate_legend:  
        fig_legend = matplotlib.pyplot.figure(figsize=(25, 2)) 
        fig_legend.legend([ideal_repro_line[0]] + list(model_points.values()) + [fit_line[0]],  
                          ['y = x'] + list(model_points.keys()) + ['Linear fit'],   
                          fontsize=label_fontsize, ncol=num_legend_columns, markerscale=1.5, 
                          loc='center', frameon=False) 
        fig_legend.tight_layout()    
        return fig, ax, fig_legend  

    elif return_ordered_names:
        return fig, ax, model_names

    else:
        return fig, ax


def model_scatter_plot_hyp(df, x_axis, y_axis, xlim, ylim, model_types,
                       title, transform='logit', x_unit='top-1, %', y_unit='top-1, %', include_legend=True,
                       tick_multiplier=10, extra_x_ticks=[], extra_y_ticks=[], set_aspect=False,
                       return_ordered_names=False, x_label=None, y_label=None, figsize=(10, 10), 
                       include_ideal=True, fit_color="red", bl_value=None, 
                       bl_name=None, return_separate_legend=False, num_legend_columns=3, error_line_width=0.5, error_alpha=0.5,
                       alpha=0.5, x_tick_multiplier=None, y_tick_multiplier=None, arrow_params=None):

    assert (df[df.show_in_plot == True][x_axis] <= xlim[1]).all()
    assert (df[df.show_in_plot == True][x_axis] >= xlim[0]).all()
    assert (df[df.show_in_plot == True][y_axis] <= ylim[1]).all()
    assert (df[df.show_in_plot == True][y_axis] >= ylim[0]).all()
    print('ylim range / xlim range aspect ratio: ', (ylim[1] - ylim[0]) / (xlim[1] - xlim[0]))

    fig, ax = matplotlib.pyplot.subplots(1, figsize=figsize)
    ax.grid(which='both', color='lightgray', linestyle='-', linewidth=grid_linewidth)

    x_tick_multiplier = x_tick_multiplier if x_tick_multiplier is not None else tick_multiplier
    y_tick_multiplier = y_tick_multiplier if y_tick_multiplier is not None else tick_multiplier
    tick_loc_x = np.array(sorted(tick_locs(xlim[0], xlim[1], x_tick_multiplier) + extra_x_ticks))
    ax.set_xticks(transform_acc(tick_loc_x, transform))
    ax.set_xticklabels([str(int(loc)) for loc in tick_loc_x])
    tick_loc_y = np.array(sorted(tick_locs(ylim[0], ylim[1], y_tick_multiplier) + extra_y_ticks))
    ax.set_yticks(transform_acc(tick_loc_y, transform))
    ax.set_yticklabels([str(int(loc)) for loc in tick_loc_y])
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    df_line = df[df.use_for_line_fit == True]
    x_acc_line_trans = transform_acc(df_line[x_axis], transform)
    y_acc_line_trans = transform_acc(df_line[y_axis], transform)

    xs = np.linspace(transform_acc(xlim[0], transform), transform_acc(xlim[1], transform), x_plotting_resolution)
    lin_fit = scipy.stats.linregress(x_acc_line_trans, y_acc_line_trans)
    slope = lin_fit[0]
    intercept = lin_fit[1]
    lin_fit_ys = xs * slope + intercept
    idx = np.argmax(y_acc_line_trans)
    x_max = x_acc_line_trans[idx]
    y_max = y_acc_line_trans[idx]
    print("ymax",y_max)
    random_slope = y_max/x_max

    print(f'Slope {slope}, intercept {intercept}, r {lin_fit[2]}, pvalue {lin_fit[3]}, stderr {lin_fit[4]}')

    ax.set_xlim([transform_acc(xlim[0], transform), transform_acc(xlim[1], transform)])
    ax.set_ylim([transform_acc(ylim[0], transform), transform_acc(ylim[1], transform)])
    if include_ideal:
        ideal_repro_line = ax.plot(xs, xs, linestyle='dashed', color='black', linewidth=main_linewidth, label='y = x')

    df_plot = df[df.show_in_plot == True]
    x_acc_plot = df_plot[x_axis]
    y_acc_plot = df_plot[y_axis]
    x_acc_ci = pd.DataFrame(df_plot[f'{x_axis}_ci'].tolist(), index=df_plot.index).to_numpy()
    y_acc_ci = pd.DataFrame(df_plot[f'{y_axis}_ci'].tolist(), index=df_plot.index).to_numpy()
    x_acc_plot_trans = transform_acc(x_acc_plot, transform)
    y_acc_plot_trans = transform_acc(y_acc_plot, transform)
    labels = df_plot.model_type.to_numpy()

    xerr_low = x_acc_plot_trans - transform_acc(x_acc_plot - x_acc_ci[:, 0], transform)
    xerr_high = transform_acc(x_acc_plot + x_acc_ci[:, 1], transform) - x_acc_plot_trans
    xerr = np.stack((xerr_low, xerr_high), axis=0)
    yerr_low = y_acc_plot_trans - transform_acc(y_acc_plot - y_acc_ci[:, 0], transform)
    yerr_high = transform_acc(y_acc_plot + y_acc_ci[:, 1], transform) - y_acc_plot_trans
    yerr = np.stack((yerr_low, yerr_high), axis=0)

    model_points = OrderedDict()
    model_names = []
    for m in model_types:
        if len(m.value) > 2:
            n, c, s = m.value
        else:
            n, c = m.value
            s = defaultmarkersize
        if not any(labels == m):
            continue
        points = ax.scatter(x_acc_plot_trans[labels == m], y_acc_plot_trans[labels == m],
                            zorder=9, color=c, s=s, label=n, alpha=alpha, linewidths=0,
                            marker='*' if c == 'tab:green' else 'o')
        model_points[n] = points
        # Reverse models within groups, then reverse all models. This is the
        # path that gives us the order that matches matplotlib's artists.
        model_names.extend(list(reversed(df_plot[labels == m].index)))
    model_names = model_names[::-1]

    fit_line = ax.plot(xs, lin_fit_ys, color=f'tab:{fit_color}', zorder=7, linewidth=main_linewidth, label=r'Baseline accuracy', alpha=0.7)

    arrow_params = [transform_acc(x, transform) if i < 2 else x for i, x in enumerate(arrow_params)]
    ax.arrow(*arrow_params, zorder=8, head_width=0.015, color='tab:green', alpha=0.7)
    ax.text(arrow_params[0]-0.3, arrow_params[1]+0.07, 'Effective\nRobustness', color='tab:green', alpha=0.9, size=25)

    ax.set_xlabel(f'{x_label if x_label else x_axis} ({x_unit})' , fontsize=label_fontsize)
    ax.set_ylabel(f'{y_label if y_label else y_axis} ({y_unit})', fontsize=label_fontsize)
    ax.set_title(f'{title}', fontsize=label_fontsize)
    if set_aspect:
        ax.set_aspect('equal', adjustable='box')
    if include_legend:
        if not include_ideal:
            ax.legend(list(model_points.values()) + [fit_line[0]],
                    list(model_points.keys()) + [r'Baseline accuracy'],
                    fontsize=label_fontsize, markerscale=1.5, frameon=False)
        else:
            ax.legend([ideal_repro_line[0]] + list(model_points.values()) + [fit_line[0]],
                    ['y = x'] + list(model_points.keys()) + [r'Baseline accuracy'],
                    fontsize=label_fontsize, markerscale=1.5, frameon=False)
    fig.tight_layout()

    if return_separate_legend:  
        fig_legend = matplotlib.pyplot.figure(figsize=(25, 2)) 
        fig_legend.legend([ideal_repro_line[0]] + list(model_points.values()) + [fit_line[0]],  
                          ['y = x'] + list(model_points.keys()) + [r'Baseline accuracy'],   
                          fontsize=label_fontsize, ncol=num_legend_columns, markerscale=1.5, 
                          loc='center', frameon=False) 
        fig_legend.tight_layout()    
        return fig, ax, fig_legend  

    elif return_ordered_names:
        return fig, ax, model_names

    else:
        return fig, ax


def model_scatter_plot_bare(df, x_axis, y_axis, xlim, ylim, model_types,
                       title, transform='logit', x_unit='top-1, %', y_unit='top-1, %', include_legend=True,
                       tick_multiplier=10, extra_x_ticks=[], extra_y_ticks=[], set_aspect=False,
                       return_ordered_names=False, x_label=None, y_label=None, figsize=(10, 10), 
                        include_ideal=True, fit_color="red", bl_value=None, 
                       bl_name=None, return_separate_legend=False, num_legend_columns=3, error_line_width=0.5, error_alpha=0.5,
                       alpha=0.5, x_tick_multiplier=None, y_tick_multiplier=None, arrow_params=None):

    assert (df[df.show_in_plot == True][x_axis] <= xlim[1]).all()
    assert (df[df.show_in_plot == True][x_axis] >= xlim[0]).all()
    assert (df[df.show_in_plot == True][y_axis] <= ylim[1]).all()
    assert (df[df.show_in_plot == True][y_axis] >= ylim[0]).all()
    print('ylim range / xlim range aspect ratio: ', (ylim[1] - ylim[0]) / (xlim[1] - xlim[0]))

    fig, ax = matplotlib.pyplot.subplots(1, figsize=figsize)
    ax.grid(which='both', color='lightgray', linestyle='-', linewidth=grid_linewidth)

    x_tick_multiplier = x_tick_multiplier if x_tick_multiplier is not None else tick_multiplier
    y_tick_multiplier = y_tick_multiplier if y_tick_multiplier is not None else tick_multiplier
    tick_loc_x = np.array(sorted(tick_locs(xlim[0], xlim[1], x_tick_multiplier) + extra_x_ticks))
    ax.set_xticks(transform_acc(tick_loc_x, transform))
    ax.set_xticklabels([str(int(loc)) for loc in tick_loc_x])
    tick_loc_y = np.array(sorted(tick_locs(ylim[0], ylim[1], y_tick_multiplier) + extra_y_ticks))
    ax.set_yticks(transform_acc(tick_loc_y, transform))
    ax.set_yticklabels([str(int(loc)) for loc in tick_loc_y])
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    df_line = df[df.use_for_line_fit == True]
    x_acc_line_trans = transform_acc(df_line[x_axis], transform)
    y_acc_line_trans = transform_acc(df_line[y_axis], transform)

    xs = np.linspace(transform_acc(xlim[0], transform), transform_acc(xlim[1], transform), x_plotting_resolution)
    lin_fit = scipy.stats.linregress(x_acc_line_trans, y_acc_line_trans)
    slope = lin_fit[0]
    intercept = lin_fit[1]
    lin_fit_ys = xs * slope + intercept
    idx = np.argmax(y_acc_line_trans)
    x_max = x_acc_line_trans[idx]
    y_max = y_acc_line_trans[idx]
    print("ymax",y_max)
    random_slope = y_max/x_max

    print(f'Slope {slope}, intercept {intercept}, r {lin_fit[2]}, pvalue {lin_fit[3]}, stderr {lin_fit[4]}')

    ax.set_xlim([transform_acc(xlim[0], transform), transform_acc(xlim[1], transform)])
    ax.set_ylim([transform_acc(ylim[0], transform), transform_acc(ylim[1], transform)])
    if include_ideal:
        ideal_repro_line = ax.plot(xs, xs, linestyle='dashed', color='black', linewidth=main_linewidth, label='y = x')

    df_plot = df[df.show_in_plot == True]
    x_acc_plot = df_plot[x_axis]
    y_acc_plot = df_plot[y_axis]
    x_acc_ci = pd.DataFrame(df_plot[f'{x_axis}_ci'].tolist(), index=df_plot.index).to_numpy()
    y_acc_ci = pd.DataFrame(df_plot[f'{y_axis}_ci'].tolist(), index=df_plot.index).to_numpy()
    x_acc_plot_trans = transform_acc(x_acc_plot, transform)
    y_acc_plot_trans = transform_acc(y_acc_plot, transform)
    labels = df_plot.model_type.to_numpy()

    xerr_low = x_acc_plot_trans - transform_acc(x_acc_plot - x_acc_ci[:, 0], transform)
    xerr_high = transform_acc(x_acc_plot + x_acc_ci[:, 1], transform) - x_acc_plot_trans
    xerr = np.stack((xerr_low, xerr_high), axis=0)
    yerr_low = y_acc_plot_trans - transform_acc(y_acc_plot - y_acc_ci[:, 0], transform)
    yerr_high = transform_acc(y_acc_plot + y_acc_ci[:, 1], transform) - y_acc_plot_trans
    yerr = np.stack((yerr_low, yerr_high), axis=0)

    model_points = OrderedDict()
    model_names = []
    for m in model_types:
        if len(m.value) > 2:
            n, c, s, a, marker = m.value
        if not any(labels == m):
            continue
        points = ax.scatter(x_acc_plot_trans[labels == m], y_acc_plot_trans[labels == m],
                            zorder=9, color=c, s=s, label=n, alpha=a, linewidths=0, marker=marker)
        model_points[n] = points
        # Reverse models within groups, then reverse all models. This is the
        # path that gives us the order that matches matplotlib's artists.
        model_names.extend(list(reversed(df_plot[labels == m].index)))
    model_names = model_names[::-1]

    fit_line = ax.plot(xs, lin_fit_ys, color=f'tab:{fit_color}', zorder=7, linewidth=main_linewidth, label=r'Baseline accuracy', alpha=0.7)

    # arrow_params = [transform_acc(x, transform) if i < 2 else x for i, x in enumerate(arrow_params)]
    # ax.arrow(*arrow_params, zorder=8, head_width=0.015, color='tab:green', alpha=0.7)
    # ax.text(arrow_params[0]+0.025, arrow_params[1]+0.18, r'$\rho$', color='tab:green', alpha=0.9, size=20)

    ax.set_xlabel(f'{x_label if x_label else x_axis} ({x_unit})' , fontsize=label_fontsize)
    ax.set_ylabel(f'{y_label if y_label else y_axis} ({y_unit})', fontsize=label_fontsize)
    ax.set_title(f'{title}', fontsize=label_fontsize)
    if set_aspect:
        ax.set_aspect('equal', adjustable='box')
    if include_legend:
        if not include_ideal:
            ax.legend(list(model_points.values()) + [fit_line[0]],
                    list(model_points.keys()) + [r'Baseline accuracy'],
                    fontsize=label_fontsize, markerscale=1.5, frameon=False)
        else:
            ax.legend([ideal_repro_line[0]] + list(model_points.values()) + [fit_line[0]],
                    ['y = x'] + list(model_points.keys()) + [r'Baseline accuracy'],
                    fontsize=label_fontsize, markerscale=1.5, frameon=False)
    fig.tight_layout()

    if return_separate_legend:  
        fig_legend = matplotlib.pyplot.figure(figsize=(25, 2)) 
        fig_legend.legend([ideal_repro_line[0]] + list(model_points.values()) + [fit_line[0]],  
                          ['y = x'] + list(model_points.keys()) + [r'Baseline accuracy'],   
                          fontsize=label_fontsize, ncol=num_legend_columns, markerscale=1.5, 
                          loc='center', frameon=False) 
        fig_legend.tight_layout()    
        return fig, ax, fig_legend  

    elif return_ordered_names:
        return fig, ax, model_names

    else:
        return fig, ax


def simple_scatter_plot(df, x_axis, y_axis, xlim, ylim, model_types,
                       title, include_legend=True,
                       tick_multiplier=10, extra_x_ticks=[], extra_y_ticks=[], set_aspect=False,
                       return_ordered_names=False, x_label=None, y_label=None, figsize=(10, 10), 
                       return_separate_legend=False, num_legend_columns=3,
                       alpha=0.7, x_tick_multiplier=None, y_tick_multiplier=None):

    assert (df[df.show_in_plot == True][x_axis] <= xlim[1]).all()
    assert (df[df.show_in_plot == True][x_axis] >= xlim[0]).all()
    assert (df[df.show_in_plot == True][y_axis] <= ylim[1]).all()
    assert (df[df.show_in_plot == True][y_axis] >= ylim[0]).all()
    print('ylim range / xlim range aspect ratio: ', (ylim[1] - ylim[0]) / (xlim[1] - xlim[0]))

    fig, ax = matplotlib.pyplot.subplots(1, figsize=figsize)
    ax.grid(which='both', color='lightgray', linestyle='-', linewidth=grid_linewidth)

    x_tick_multiplier = x_tick_multiplier if x_tick_multiplier is not None else tick_multiplier
    y_tick_multiplier = y_tick_multiplier if y_tick_multiplier is not None else tick_multiplier
    tick_loc_x = np.array(sorted(tick_locs(xlim[0], xlim[1], x_tick_multiplier) + extra_x_ticks))
    ax.set_xticks(tick_loc_x)
    ax.set_xticklabels([str(int(loc)) for loc in tick_loc_x])
    tick_loc_y = np.array(sorted(tick_locs(ylim[0], ylim[1], y_tick_multiplier) + extra_y_ticks))
    ax.set_yticks(tick_loc_y)
    ax.set_yticklabels([str(int(loc)) for loc in tick_loc_y])
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    df_plot = df[df.show_in_plot == True]
    x_acc_plot = df_plot[x_axis]
    y_acc_plot = df_plot[y_axis]
    labels = df_plot.model_type.to_numpy()

    model_points = OrderedDict()
    model_names = []
    for m in model_types:
        if len(m.value) > 2:
            n, c, s = m.value
        else:
            n, c = m.value
            s = defaultmarkersize
        if not any(labels == m):
            continue
        # alpha = None if isinstance(c, (list, tuple)) and len(c) == 4 else 0.5
        points = ax.scatter(x_acc_plot[labels == m], y_acc_plot[labels == m],
                            zorder=9, color=c, s=s, label=n, alpha=alpha, linewidths=0)
        model_points[n] = points
        # Reverse models within groups, then reverse all models. This is the
        # path that gives us the order that matches matplotlib's artists.
        model_names.extend(list(reversed(df_plot[labels == m].index)))
    model_names = model_names[::-1]

    ax.set_xlabel(f'{x_label if x_label else x_axis}' , fontsize=label_fontsize)
    ax.set_ylabel(f'{y_label if y_label else y_axis}', fontsize=label_fontsize)
    ax.set_title(f'{title}', fontsize=label_fontsize)

    if set_aspect:
        ax.set_aspect('equal', adjustable='box')

    if include_legend:
        ax.legend(list(model_points.values()),
                list(model_points.keys()),
                fontsize=label_fontsize, markerscale=1.5, frameon=False)
    fig.tight_layout()

    if return_separate_legend:  
        fig_legend = matplotlib.pyplot.figure(figsize=(25, 2)) 
        fig_legend.legend(list(model_points.values()),
                          list(model_points.keys()),   
                          fontsize=label_fontsize, ncol=num_legend_columns, markerscale=1.5, 
                          loc='center', frameon=False) 
        fig_legend.tight_layout()    
        return fig, ax, fig_legend  

    elif return_ordered_names:
        return fig, ax, model_names

    else:
        return fig, ax


def model_scatter_plot_quadrants(df, x_axis, y_axis, xlim, ylim, model_types, num_bootstrap_samples,
                       title, transform='logit', x_unit='top-1, %', y_unit='top-1, %', include_legend=True,
                       tick_multiplier=10, extra_x_ticks=[], extra_y_ticks=[], set_aspect=False,
                       return_ordered_names=False, x_label=None, y_label=None, figsize=(10, 10), 
                       include_ideal=True, fit_color="red", bl_value=None, 
                       bl_name=None, return_separate_legend=False, num_legend_columns=3, error_line_width=0.5, error_alpha=0.5,
                       alpha=0.5):

    assert (df[df.show_in_plot == True][x_axis] <= xlim[1]).all()
    assert (df[df.show_in_plot == True][x_axis] >= xlim[0]).all()
    assert (df[df.show_in_plot == True][y_axis] <= ylim[1]).all()
    assert (df[df.show_in_plot == True][y_axis] >= ylim[0]).all()
    print('ylim range / xlim range aspect ratio: ', (ylim[1] - ylim[0]) / (xlim[1] - xlim[0]))

    fig, ax = matplotlib.pyplot.subplots(1, figsize=figsize)
    ax.grid(which='both', color='lightgray', linestyle='-', linewidth=grid_linewidth)

    tick_loc_x = np.array(sorted(tick_locs(xlim[0], xlim[1], tick_multiplier) + extra_x_ticks))
    ax.set_xticks(transform_acc(tick_loc_x, transform))
    ax.set_xticklabels([str(int(loc)) for loc in tick_loc_x])
    tick_loc_y = np.array(sorted(tick_locs(ylim[0], ylim[1], tick_multiplier) + extra_y_ticks))
    ax.set_yticks(transform_acc(tick_loc_y, transform))
    ax.set_yticklabels([str(int(loc)) for loc in tick_loc_y])
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    df_line = df[df.use_for_line_fit == True]
    x_acc_line_trans = transform_acc(df_line[x_axis], transform)
    y_acc_line_trans = transform_acc(df_line[y_axis], transform)

    xs = np.linspace(transform_acc(xlim[0], transform), transform_acc(xlim[1], transform), x_plotting_resolution)
    lin_fit = scipy.stats.linregress(x_acc_line_trans, y_acc_line_trans)
    slope = lin_fit[0]
    intercept = lin_fit[1]
    lin_fit_ys = xs * slope + intercept
    idx = np.argmax(y_acc_line_trans)
    x_max = x_acc_line_trans[idx]
    y_max = y_acc_line_trans[idx]
    print("ymax",y_max)
    random_slope = y_max/x_max

    print(f'Slope {slope}, intercept {intercept}, r {lin_fit[2]}, pvalue {lin_fit[3]}, stderr {lin_fit[4]}')
    coeffs_ci, fit_lower, fit_upper = get_bootstrap_cis(x_acc_line_trans, y_acc_line_trans, num_bootstrap_samples, xs, 720257663,
                                                        significance_level_coeffs=95, significance_level_grid=95)
    print(f'Bootstrap CIs: {coeffs_ci}')
    sm_model = sm.OLS(y_acc_line_trans, np.stack([np.ones(x_acc_line_trans.shape[0]), x_acc_line_trans], axis=1))
    sm_results = sm_model.fit()
    print(sm_results.summary())

    ax.set_xlim([transform_acc(xlim[0], transform), transform_acc(xlim[1], transform)])
    ax.set_ylim([transform_acc(ylim[0], transform), transform_acc(ylim[1], transform)])
    if include_ideal:
        df_small = df[df.model_type == model_types.STANDARD][df.show_in_plot == True]
        line = np.linspace(transform_acc(df_small[y_axis][0], transform), transform_acc(df_small[y_axis][0], transform), x_plotting_resolution)
        ideal_repro_line = ax.plot(xs, line, linestyle='dotted', color='black', linewidth=main_linewidth, label=f'ResNet50 baseline')
    # ax.fill_between(xs, fit_upper, fit_lower, color=f'tab:{fit_color}', alpha=0.3, zorder=6, edgecolor='none', linewidth=2.0)

    df_plot = df[df.show_in_plot == True]
    x_acc_plot = df_plot[x_axis]
    y_acc_plot = df_plot[y_axis]
    x_acc_ci = pd.DataFrame(df_plot[f'{x_axis}_ci'].tolist(), index=df_plot.index).to_numpy()
    y_acc_ci = pd.DataFrame(df_plot[f'{y_axis}_ci'].tolist(), index=df_plot.index).to_numpy()
    x_acc_plot_trans = transform_acc(x_acc_plot, transform)
    y_acc_plot_trans = transform_acc(y_acc_plot, transform)
    labels = df_plot.model_type.to_numpy()

    xerr_low = x_acc_plot_trans - transform_acc(x_acc_plot - x_acc_ci[:, 0], transform)
    xerr_high = transform_acc(x_acc_plot + x_acc_ci[:, 1], transform) - x_acc_plot_trans
    xerr = np.stack((xerr_low, xerr_high), axis=0)
    yerr_low = y_acc_plot_trans - transform_acc(y_acc_plot - y_acc_ci[:, 0], transform)
    yerr_high = transform_acc(y_acc_plot + y_acc_ci[:, 1], transform) - y_acc_plot_trans
    yerr = np.stack((yerr_low, yerr_high), axis=0)

    model_points = OrderedDict()
    model_names = []
    for m in model_types:
        if len(m.value) > 2:
            n, c, s = m.value
        else:
            n, c = m.value
            s = defaultmarkersize
        if not any(labels == m):
            continue
        ax.errorbar(x_acc_plot_trans[labels == m], y_acc_plot_trans[labels == m], xerr=xerr[:, labels == m], yerr=yerr[:, labels == m],
                    capsize=2, linewidth=error_line_width, ls='none', color=c, alpha=error_alpha, zorder=8)
        # alpha = None if isinstance(c, (list, tuple)) and len(c) == 4 else 0.5
        points = ax.scatter(x_acc_plot_trans[labels == m], y_acc_plot_trans[labels == m],
                            zorder=9, color=c, s=s, label=n, alpha=alpha, linewidths=0)
        model_points[n] = points
        # Reverse models within groups, then reverse all models. This is the
        # path that gives us the order that matches matplotlib's artists.
        model_names.extend(list(reversed(df_plot[labels == m].index)))
    model_names = model_names[::-1]

    fit_line = ax.plot(xs, lin_fit_ys, color=f'tab:{fit_color}', zorder=7, linewidth=main_linewidth, label='Linear fit', alpha=0.5)

    bottom = np.linspace(transform_acc(ylim[0], transform), transform_acc(ylim[0], transform), x_plotting_resolution)
    poor_rel = ax.fill_between(xs, line, bottom, color=f'black', zorder=5, alpha=0.1)
    poor_rel = ax.fill_between(xs, line, bottom, where=[False]*len(xs), color=f'black', zorder=5, alpha=0.3, label='Negative relative robustness')
    poor_eff = ax.fill_between(xs, lin_fit_ys, bottom, color=f'tab:{fit_color}', zorder=5, alpha=0.1)
    poor_eff = ax.fill_between(xs, lin_fit_ys, bottom, where=[False]*len(xs), color=f'tab:{fit_color}', zorder=5, alpha=0.3, label='Negative effective robustness')

    ax.set_xlabel(f'{x_label if x_label else x_axis} ({x_unit})' , fontsize=label_fontsize)
    ax.set_ylabel(f'{y_label if y_label else y_axis} ({y_unit})', fontsize=label_fontsize)
    ax.set_title(f'{title}', fontsize=label_fontsize)
    if set_aspect:
        ax.set_aspect('equal', adjustable='box')
    if include_legend:
        if not include_ideal:
            ax.legend(list(model_points.values()) + [fit_line[0]],
                    list(model_points.keys()) + ['Linear fit'],
                    fontsize=label_fontsize, markerscale=1.5, frameon=False)
        else:
            ax.legend(list(model_points.values()) + [ideal_repro_line[0]] + [fit_line[0]],
                    list(model_points.keys()) + [f'ResNet50 baseline'] + ['Linear fit'],
                    fontsize=label_fontsize, markerscale=1.5, frameon=False)
    fig.tight_layout()

    if return_separate_legend:  
        fig_legend = matplotlib.pyplot.figure(figsize=(25, 2)) 
        poor_rel = fig_legend.gca().fill_between(xs, line, bottom, where=[False]*len(xs), color=f'black', zorder=5, alpha=0.3, label='Negative relative robustness')
        poor_eff = fig_legend.gca().fill_between(xs, lin_fit_ys, bottom, where=[False]*len(xs), color=f'tab:{fit_color}', zorder=5, alpha=0.3, label='Negative effective robustness')
        fig_legend.gca().axis('off')
        fig_legend.legend(
                          [ideal_repro_line[0]] + list(model_points.values()) + [fit_line[0], poor_rel, poor_eff],  
                          [f'ResNet50 baseline'] + list(model_points.keys()) + ['Linear fit', 'Negative relative robustness', 'Negative effective robustness'],   
                          fontsize=label_fontsize, ncol=num_legend_columns, markerscale=1.5, 
                          loc='center', frameon=False) 
        fig_legend.tight_layout(pad=1.0)
        return fig, ax, fig_legend  

    elif return_ordered_names:
        return fig, ax, model_names

    else:
        return fig, ax


def model_scatter_plot_quadrants_imagenet_a(df, x_axis, y_axis, xlim, ylim, model_types, num_bootstrap_samples,
                       title, transform='logit', x_unit='top-1, %', y_unit='top-1, %', include_legend=True,
                       tick_multiplier=10, extra_x_ticks=[], extra_y_ticks=[], set_aspect=False,
                       return_ordered_names=False, x_label=None, y_label=None, figsize=(10, 10), 
                       include_ideal=True, fit_color="red", bl_value=None, 
                       bl_name=None, return_separate_legend=False, num_legend_columns=3, error_line_width=0.5, error_alpha=0.5,
                       alpha=0.5, pivot=None):

    assert (df[df.show_in_plot == True][x_axis] <= xlim[1]).all()
    assert (df[df.show_in_plot == True][x_axis] >= xlim[0]).all()
    assert (df[df.show_in_plot == True][y_axis] <= ylim[1]).all()
    assert (df[df.show_in_plot == True][y_axis] >= ylim[0]).all()
    print('ylim range / xlim range aspect ratio: ', (ylim[1] - ylim[0]) / (xlim[1] - xlim[0]))

    fig, ax = matplotlib.pyplot.subplots(1, figsize=figsize)
    ax.grid(which='both', color='lightgray', linestyle='-', linewidth=grid_linewidth)

    tick_loc_x = np.array(sorted(tick_locs(xlim[0], xlim[1], tick_multiplier) + extra_x_ticks))
    ax.set_xticks(transform_acc(tick_loc_x, transform))
    ax.set_xticklabels([str(int(loc)) for loc in tick_loc_x])
    tick_loc_y = np.array(sorted(tick_locs(ylim[0], ylim[1], tick_multiplier) + extra_y_ticks))
    ax.set_yticks(transform_acc(tick_loc_y, transform))
    ax.set_yticklabels([str(int(loc)) for loc in tick_loc_y])
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    
    xs = np.linspace(transform_acc(xlim[0], transform), transform_acc(xlim[1], transform), x_plotting_resolution)
    ys = np.linspace(transform_acc(ylim[0], transform), transform_acc(ylim[1], transform), x_plotting_resolution)
    ax.set_xlim([transform_acc(xlim[0], transform), transform_acc(xlim[1], transform)])
    ax.set_ylim([transform_acc(ylim[0], transform), transform_acc(ylim[1], transform)])
    if include_ideal:
        df_small = df[df.model_type == model_types.STANDARD][df.show_in_plot == True]
        line = np.linspace(transform_acc(df_small[y_axis][0], transform), transform_acc(df_small[y_axis][0], transform), x_plotting_resolution)
        ideal_repro_line = ax.plot(xs, line, linestyle='dotted', color='black', linewidth=main_linewidth, label=f'ResNet50 baseline')
    
    pivot_trans = transform_acc(pivot, transform)

    df_line = df[(df.use_for_line_fit == True) & (df[x_axis] < pivot)]
    x_acc_line_trans = transform_acc(df_line[x_axis], transform)
    y_acc_line_trans = transform_acc(df_line[y_axis], transform)
    lin_fit = scipy.stats.linregress(x_acc_line_trans, y_acc_line_trans)
    slope = lin_fit[0]
    intercept = lin_fit[1]
    lin_fit_ys1 = xs * slope + intercept
    idx = np.argmax(y_acc_line_trans)
    x_max = x_acc_line_trans[idx]
    y_max = y_acc_line_trans[idx]
    print("ymax",y_max)
    random_slope = y_max/x_max

    print(f'Slope {slope}, intercept {intercept}, r {lin_fit[2]}, pvalue {lin_fit[3]}, stderr {lin_fit[4]}')
    coeffs_ci, fit_lower, fit_upper = get_bootstrap_cis(x_acc_line_trans, y_acc_line_trans, num_bootstrap_samples, xs, 720257663,
                                                        significance_level_coeffs=95, significance_level_grid=95)
    print(f'Bootstrap CIs: {coeffs_ci}')
    sm_model = sm.OLS(y_acc_line_trans, np.stack([np.ones(x_acc_line_trans.shape[0]), x_acc_line_trans], axis=1))
    sm_results = sm_model.fit()
    print(sm_results.summary())

    # ax.fill_between(xs, fit_upper, fit_lower, where=xs<pivot_trans, color=f'tab:{fit_color}', alpha=0.3, zorder=6, edgecolor='none', linewidth=2.0)
    fit_line_one = ax.plot(xs[xs<pivot_trans], lin_fit_ys1[xs<pivot_trans], color=f'tab:{fit_color}', zorder=7, linewidth=main_linewidth, alpha=0.5)

    df_line = df[(df.use_for_line_fit == True) & (df[x_axis] > pivot)]
    x_acc_line_trans = transform_acc(df_line[x_axis], transform)
    y_acc_line_trans = transform_acc(df_line[y_axis], transform)
    lin_fit = scipy.stats.linregress(x_acc_line_trans, y_acc_line_trans)
    slope = lin_fit[0]
    intercept = lin_fit[1]
    lin_fit_ys2 = xs * slope + intercept
    idx = np.argmax(y_acc_line_trans)
    x_max = x_acc_line_trans[idx]
    y_max = y_acc_line_trans[idx]
    print("ymax",y_max)
    random_slope = y_max/x_max

    print(f'Slope {slope}, intercept {intercept}, r {lin_fit[2]}, pvalue {lin_fit[3]}, stderr {lin_fit[4]}')
    coeffs_ci, fit_lower, fit_upper = get_bootstrap_cis(x_acc_line_trans, y_acc_line_trans, num_bootstrap_samples, xs, 720257663,
                                                        significance_level_coeffs=95, significance_level_grid=95)
    print(f'Bootstrap CIs: {coeffs_ci}')
    sm_model = sm.OLS(y_acc_line_trans, np.stack([np.ones(x_acc_line_trans.shape[0]), x_acc_line_trans], axis=1))
    sm_results = sm_model.fit()
    print(sm_results.summary())

    # ax.fill_between(xs, fit_upper, fit_lower, where=xs>=pivot_trans, color=f'tab:{fit_color}', alpha=0.3, zorder=6, edgecolor='none', linewidth=2.0)
    fit_line_two = ax.plot(xs[xs>pivot_trans], lin_fit_ys2[xs>=pivot_trans], color=f'tab:{fit_color}', zorder=7, linewidth=main_linewidth, label='Linear fit (piecewise)', alpha=0.5)

    resnet50 = np.linspace(pivot_trans, pivot_trans, x_plotting_resolution)
    resnet50_line = ax.plot(resnet50, ys, linestyle='dashed', color='black', linewidth=main_linewidth//2, label='ResNet50 accuracy')


    df_plot = df[df.show_in_plot == True]
    x_acc_plot = df_plot[x_axis]
    y_acc_plot = df_plot[y_axis]
    x_acc_ci = pd.DataFrame(df_plot[f'{x_axis}_ci'].tolist(), index=df_plot.index).to_numpy()
    y_acc_ci = pd.DataFrame(df_plot[f'{y_axis}_ci'].tolist(), index=df_plot.index).to_numpy()
    x_acc_plot_trans = transform_acc(x_acc_plot, transform)
    y_acc_plot_trans = transform_acc(y_acc_plot, transform)
    labels = df_plot.model_type.to_numpy()

    xerr_low = x_acc_plot_trans - transform_acc(x_acc_plot - x_acc_ci[:, 0], transform)
    xerr_high = transform_acc(x_acc_plot + x_acc_ci[:, 1], transform) - x_acc_plot_trans
    xerr = np.stack((xerr_low, xerr_high), axis=0)
    yerr_low = y_acc_plot_trans - transform_acc(y_acc_plot - y_acc_ci[:, 0], transform)
    yerr_high = transform_acc(y_acc_plot + y_acc_ci[:, 1], transform) - y_acc_plot_trans
    yerr = np.stack((yerr_low, yerr_high), axis=0)

    model_points = OrderedDict()
    model_names = []
    for m in model_types:
        if len(m.value) > 2:
            n, c, s = m.value
        else:
            n, c = m.value
            s = defaultmarkersize
        if not any(labels == m):
            continue
        ax.errorbar(x_acc_plot_trans[labels == m], y_acc_plot_trans[labels == m], xerr=xerr[:, labels == m], yerr=yerr[:, labels == m],
                    capsize=2, linewidth=error_line_width, ls='none', color=c, alpha=error_alpha, zorder=8)
        # alpha = None if isinstance(c, (list, tuple)) and len(c) == 4 else 0.5
        points = ax.scatter(x_acc_plot_trans[labels == m], y_acc_plot_trans[labels == m],
                            zorder=9, color=c, s=s, label=n, alpha=alpha, linewidths=0)
        model_points[n] = points
        # Reverse models within groups, then reverse all models. This is the
        # path that gives us the order that matches matplotlib's artists.
        model_names.extend(list(reversed(df_plot[labels == m].index)))
    model_names = model_names[::-1]

    bottom = np.linspace(transform_acc(ylim[0], transform), transform_acc(ylim[0], transform), x_plotting_resolution)
    poor_rel = ax.fill_between(xs, line, bottom, color=f'black', zorder=5, alpha=0.1)
    poor_rel = ax.fill_between(xs, line, bottom, where=[False]*len(xs), color=f'black', zorder=5, alpha=0.3, label='Negative relative robustness')
    poor_eff = ax.fill_between(xs, lin_fit_ys1, bottom, where=xs<=pivot_trans, color=f'tab:{fit_color}', zorder=5, alpha=0.1)
    poor_eff = ax.fill_between(xs, lin_fit_ys2, bottom, where=xs>=pivot_trans, color=f'tab:{fit_color}', zorder=5, alpha=0.1)
    poor_eff = ax.fill_between(xs, lin_fit_ys2, bottom, where=[False]*len(xs), color=f'tab:{fit_color}', zorder=5, alpha=0.3, label='Negative effective robustness')

    ax.set_xlabel(f'{x_label if x_label else x_axis} ({x_unit})' , fontsize=label_fontsize)
    ax.set_ylabel(f'{y_label if y_label else y_axis} ({y_unit})', fontsize=label_fontsize)
    ax.set_title(f'{title}', fontsize=label_fontsize)
    if set_aspect:
        ax.set_aspect('equal', adjustable='box')
    if include_legend:
        if not include_ideal:
            ax.legend(list(model_points.values()) + [fit_line[0]],
                    list(model_points.keys()) + ['Linear fit'],
                    fontsize=label_fontsize, markerscale=1.5, frameon=False)
        else:
            ax.legend(list(model_points.values()) + [ideal_repro_line[0]] + [fit_line[0]],
                    list(model_points.keys()) + [f'ResNet50 baseline'] + ['Linear fit'],
                    fontsize=label_fontsize, markerscale=1.5, frameon=False)
    fig.tight_layout()

    if return_separate_legend:  
        fig_legend = matplotlib.pyplot.figure(figsize=(25, 2)) 
        poor_rel = fig_legend.gca().fill_between(xs, line, bottom, where=[False]*len(xs), color=f'black', zorder=5, alpha=0.3, label='Negative relative robustness')
        poor_eff = fig_legend.gca().fill_between(xs, lin_fit_ys2, bottom, where=[False]*len(xs), color=f'tab:{fit_color}', zorder=5, alpha=0.3, label='Negative effective robustness')
        fig_legend.gca().axis('off')
        fig_legend.legend(
                          [ideal_repro_line[0]] + list(model_points.values()) + [fit_line_one[0], poor_rel, poor_eff],  
                          [f'ResNet50 baseline'] + list(model_points.keys()) + ['Linear fit (piecewise)', 'Negative relative robustness', 'Negative effective robustness'],   
                          fontsize=label_fontsize, ncol=num_legend_columns, markerscale=1.5, 
                          loc='center', frameon=False) 
        fig_legend.tight_layout(pad=1.0)
        return fig, ax, fig_legend  

    elif return_ordered_names:
        return fig, ax, model_names

    else:
        return fig, ax


def hyp_model_scatter_plot_quadrants(df, x_axis, y_axis, xlim, ylim, model_types,
                       title, transform='logit', x_unit='top-1, %', y_unit='top-1, %', include_legend=True,
                       tick_multiplier=10, extra_x_ticks=[], extra_y_ticks=[], set_aspect=False,
                       return_ordered_names=False, x_label=None, y_label=None, figsize=(10, 10), 
                       include_ideal=True, fit_color="red", bl_value=None, 
                       bl_name=None, return_separate_legend=False, num_legend_columns=3, error_line_width=0.5, error_alpha=0.5,
                       alpha=0.5, arrow_params=None):

    assert (df[df.show_in_plot == True][x_axis] <= xlim[1]).all()
    assert (df[df.show_in_plot == True][x_axis] >= xlim[0]).all()
    assert (df[df.show_in_plot == True][y_axis] <= ylim[1]).all()
    assert (df[df.show_in_plot == True][y_axis] >= ylim[0]).all()
    print('ylim range / xlim range aspect ratio: ', (ylim[1] - ylim[0]) / (xlim[1] - xlim[0]))

    fig, ax = matplotlib.pyplot.subplots(1, figsize=figsize)
    ax.grid(which='both', color='lightgray', linestyle='-', linewidth=grid_linewidth)

    tick_loc_x = np.array(sorted(tick_locs(xlim[0], xlim[1], tick_multiplier) + extra_x_ticks))
    ax.set_xticks(transform_acc(tick_loc_x, transform))
    ax.set_xticklabels([str(int(loc)) for loc in tick_loc_x])
    tick_loc_y = np.array(sorted(tick_locs(ylim[0], ylim[1], tick_multiplier) + extra_y_ticks))
    ax.set_yticks(transform_acc(tick_loc_y, transform))
    ax.set_yticklabels([str(int(loc)) for loc in tick_loc_y])
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    df_line = df[df.use_for_line_fit == True]
    x_acc_line_trans = transform_acc(df_line[x_axis], transform)
    y_acc_line_trans = transform_acc(df_line[y_axis], transform)

    xs = np.linspace(transform_acc(xlim[0], transform), transform_acc(xlim[1], transform), x_plotting_resolution)
    lin_fit = scipy.stats.linregress(x_acc_line_trans, y_acc_line_trans)
    slope = lin_fit[0]
    intercept = lin_fit[1]
    lin_fit_ys = xs * slope + intercept
    idx = np.argmax(y_acc_line_trans)
    x_max = x_acc_line_trans[idx]
    y_max = y_acc_line_trans[idx]
    print("ymax",y_max)
    random_slope = y_max/x_max

    print(f'Slope {slope}, intercept {intercept}, r {lin_fit[2]}, pvalue {lin_fit[3]}, stderr {lin_fit[4]}')
    # coeffs_ci, fit_lower, fit_upper = get_bootstrap_cis(x_acc_line_trans, y_acc_line_trans, num_bootstrap_samples, xs, 720257663,
    #                                                     significance_level_coeffs=95, significance_level_grid=95)
    # print(f'Bootstrap CIs: {coeffs_ci}')
    # sm_model = sm.OLS(y_acc_line_trans, np.stack([np.ones(x_acc_line_trans.shape[0]), x_acc_line_trans], axis=1))
    # sm_results = sm_model.fit()
    # print(sm_results.summary())

    ax.set_xlim([transform_acc(xlim[0], transform), transform_acc(xlim[1], transform)])
    ax.set_ylim([transform_acc(ylim[0], transform), transform_acc(ylim[1], transform)])
    if include_ideal:
        df_small = df[df.model_type == model_types.STANDARD][df.show_in_plot == True]
        line = np.linspace(transform_acc(df_small[y_axis][0], transform), transform_acc(df_small[y_axis][0], transform), x_plotting_resolution)
        ideal_repro_line = ax.plot(xs, line, linestyle='dotted', color='black', linewidth=main_linewidth)
    # ax.fill_between(xs, fit_upper, fit_lower, color=f'tab:{fit_color}', alpha=0.3, zorder=6, edgecolor='none', linewidth=2.0)

    df_plot = df[df.show_in_plot == True]
    x_acc_plot = df_plot[x_axis]
    y_acc_plot = df_plot[y_axis]
    x_acc_ci = pd.DataFrame(df_plot[f'{x_axis}_ci'].tolist(), index=df_plot.index).to_numpy()
    y_acc_ci = pd.DataFrame(df_plot[f'{y_axis}_ci'].tolist(), index=df_plot.index).to_numpy()
    x_acc_plot_trans = transform_acc(x_acc_plot, transform)
    y_acc_plot_trans = transform_acc(y_acc_plot, transform)
    labels = df_plot.model_type.to_numpy()

    xerr_low = x_acc_plot_trans - transform_acc(x_acc_plot - x_acc_ci[:, 0], transform)
    xerr_high = transform_acc(x_acc_plot + x_acc_ci[:, 1], transform) - x_acc_plot_trans
    xerr = np.stack((xerr_low, xerr_high), axis=0)
    yerr_low = y_acc_plot_trans - transform_acc(y_acc_plot - y_acc_ci[:, 0], transform)
    yerr_high = transform_acc(y_acc_plot + y_acc_ci[:, 1], transform) - y_acc_plot_trans
    yerr = np.stack((yerr_low, yerr_high), axis=0)

    model_points = OrderedDict()
    model_names = []
    for m in model_types:
        if len(m.value) > 2:
            n, c, s = m.value
        else:
            n, c = m.value
            s = defaultmarkersize
        if not any(labels == m):
            continue
        # ax.errorbar(x_acc_plot_trans[labels == m], y_acc_plot_trans[labels == m], xerr=xerr[:, labels == m], yerr=yerr[:, labels == m],
        #             capsize=2, linewidth=error_line_width, ls='none', color=c, alpha=error_alpha, zorder=8)
        # alpha = None if isinstance(c, (list, tuple)) and len(c) == 4 else 0.5
        points = ax.scatter(x_acc_plot_trans[labels == m], y_acc_plot_trans[labels == m],
                            zorder=9, color=c, s=s, label=n, alpha=alpha, linewidths=0)
        model_points[n] = points
        # Reverse models within groups, then reverse all models. This is the
        # path that gives us the order that matches matplotlib's artists.
        model_names.extend(list(reversed(df_plot[labels == m].index)))
    model_names = model_names[::-1]

    fit_line = ax.plot(xs, lin_fit_ys, color=f'tab:{fit_color}', zorder=3, linewidth=main_linewidth, label=r'Baseline accuracy', alpha=0.7)

    bottom = np.linspace(transform_acc(ylim[0], transform), transform_acc(ylim[0], transform), x_plotting_resolution)
    # ax.fill_between(xs, line, bottom, where=lin_fit_ys <= line[0]+0.005, color=f'black', zorder=5, alpha=0.1)
    # ax.fill_between(xs, lin_fit_ys, bottom, where=lin_fit_ys >= line[0], color=f'tab:{fit_color}', zorder=5, alpha=0.1)
    poor_rel = ax.fill_between(xs, line, bottom, color=f'black', zorder=5, alpha=0.1)
    poor_rel = ax.fill_between(xs, line, bottom, where=[False]*len(xs), color=f'black', zorder=5, alpha=0.3, label='Negative relative robustness')
    poor_eff = ax.fill_between(xs, lin_fit_ys, bottom, color=f'tab:{fit_color}', zorder=5, alpha=0.1)
    poor_eff = ax.fill_between(xs, lin_fit_ys, bottom, where=[False]*len(xs), color=f'tab:{fit_color}', zorder=5, alpha=0.3, label='Negative effective robustness')

    arrow_params = [arrow_params[0]-0.2, arrow_params[1]-0.1, arrow_params[2]+1, arrow_params[3]+0.1]
    arrow_params = [transform_acc(x, transform) for x in arrow_params]
    arrow_params[2] -= arrow_params[0]
    arrow_params[3] -= arrow_params[1]
    ax.arrow(*arrow_params, zorder=8, head_width=0.015, alpha=0.7, color='tab:green')
    # ax.text(arrow_params[0]+0.025, arrow_params[1]+0.18, r'$\rho$', color='tab:green', alpha=0.9, size=20)

    ax.set_xlabel(f'{x_label if x_label else x_axis} ({x_unit})' , fontsize=label_fontsize)
    ax.set_ylabel(f'{y_label if y_label else y_axis} ({y_unit})', fontsize=label_fontsize)
    ax.set_title(f'{title}', fontsize=label_fontsize)
    if set_aspect:
        ax.set_aspect('equal', adjustable='box')
    if include_legend:
        if not include_ideal:
            ax.legend(list(model_points.values()) + [fit_line[0]],
                    list(model_points.keys()) + [r'Baseline accuracy'],
                    fontsize=label_fontsize, markerscale=1.5, frameon=False)
        else:
            ax.legend(list(model_points.values()) + [ideal_repro_line[0], fit_line[0], poor_rel, poor_eff],
                    list(model_points.keys()) + ['ResNet50 target accuracy', r'Baseline accuracy', 'Poor relative robustness', 'Poor effective robustness'],
                    fontsize=label_fontsize, markerscale=1.5, frameon=False)
    fig.tight_layout()

    if return_separate_legend:  
        fig_legend = matplotlib.pyplot.figure(figsize=(25, 2)) 
        fig_legend.legend([ideal_repro_line[0]] + list(model_points.values()) + [fit_line[0]],  
                          ['ResNet50 target accuracy'] + list(model_points.keys()) + [r'Baseline accuracy'],   
                          fontsize=label_fontsize, ncol=num_legend_columns, markerscale=1.5, 
                          loc='center', frameon=False) 
        fig_legend.tight_layout(pad=1.0)    
        return fig, ax, fig_legend  

    elif return_ordered_names:
        return fig, ax, model_names

    else:
        return fig, ax


def get_confidence_interval(p, n, alpha=0.05, method='beta'):
    assert p >= 0.0
    assert p <= 1.0
    return statsmodels.stats.proportion.proportion_confint(p * n, n, alpha=alpha, method=method)


def add_confidence_interval_to_dataframe(df,
                                         accuracy_col_name,
                                         dataset_size_name,
                                         transform,
                                         upper_bound_name=None,
                                         lower_bound_name=None,
                                         alpha=0.05,
                                         method='beta'):
    assert accuracy_col_name in df.columns
    assert dataset_size_name in df.columns
    if upper_bound_name is None:
        upper_bound_name = accuracy_col_name + '_transformed_ci_upper_delta'
    assert upper_bound_name not in df.columns
    if lower_bound_name is None:
        lower_bound_name = accuracy_col_name + '_transformed_ci_lower_delta'
    assert lower_bound_name not in df.columns

    df2 = df.copy()
    for ii, row in df2.iterrows():
        cur_acc = row[accuracy_col_name]
        cur_n = row[dataset_size_name]
        cur_ci = get_confidence_interval(cur_acc / 100.0, cur_n, alpha=alpha)
        cur_upper_delta = transform_acc(cur_ci[1] * 100.0, transform) - transform_acc(cur_acc, transform)
        cur_lower_delta = transform_acc(cur_acc, transform) - transform_acc(cur_ci[0] * 100.0, transform)
        df2.loc[ii, upper_bound_name] = cur_upper_delta
        df2.loc[ii, lower_bound_name] = cur_lower_delta

    return df2


def model_scatter_plot_imagenet_a(df, x_axis, y_axis, xlim, ylim, model_types, num_bootstrap_samples,
                       title, transform='logit', x_unit='top-1, %', y_unit='top-1, %', include_legend=True,
                       tick_multiplier=10, extra_x_ticks=[], extra_y_ticks=[], set_aspect=False,
                       return_ordered_names=False, x_label=None, y_label=None, figsize=(10, 10), 
                        include_ideal=True, fit_color="red", bl_value=None, 
                       bl_name=None, return_separate_legend=False, num_legend_columns=3, error_line_width=0.5, error_alpha=0.5,
                       alpha=0.5, x_tick_multiplier=None, y_tick_multiplier=None, pivot=None):

    assert (df[df.show_in_plot == True][x_axis] <= xlim[1]).all()
    assert (df[df.show_in_plot == True][x_axis] >= xlim[0]).all()
    assert (df[df.show_in_plot == True][y_axis] <= ylim[1]).all()
    assert (df[df.show_in_plot == True][y_axis] >= ylim[0]).all()
    print('ylim range / xlim range aspect ratio: ', (ylim[1] - ylim[0]) / (xlim[1] - xlim[0]))

    fig, ax = matplotlib.pyplot.subplots(1, figsize=figsize)
    ax.grid(which='both', color='lightgray', linestyle='-', linewidth=grid_linewidth)

    x_tick_multiplier = x_tick_multiplier if x_tick_multiplier is not None else tick_multiplier
    y_tick_multiplier = y_tick_multiplier if y_tick_multiplier is not None else tick_multiplier
    tick_loc_x = np.array(sorted(tick_locs(xlim[0], xlim[1], x_tick_multiplier) + extra_x_ticks))
    ax.set_xticks(transform_acc(tick_loc_x, transform))
    ax.set_xticklabels([str(int(loc)) for loc in tick_loc_x])
    tick_loc_y = np.array(sorted(tick_locs(ylim[0], ylim[1], y_tick_multiplier) + extra_y_ticks))
    ax.set_yticks(transform_acc(tick_loc_y, transform))
    ax.set_yticklabels([str(int(loc)) for loc in tick_loc_y])
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)


    xs = np.linspace(transform_acc(xlim[0], transform), transform_acc(xlim[1], transform), x_plotting_resolution)
    ys = np.linspace(transform_acc(ylim[0], transform), transform_acc(ylim[1], transform), x_plotting_resolution)
    ax.set_xlim([transform_acc(xlim[0], transform), transform_acc(xlim[1], transform)])
    ax.set_ylim([transform_acc(ylim[0], transform), transform_acc(ylim[1], transform)])
    if include_ideal:
        ideal_repro_line = ax.plot(xs, xs, linestyle='dashed', color='black', linewidth=main_linewidth, label='y = x')

    pivot_trans = transform_acc(pivot, transform)

    df_line = df[df.use_for_line_fit_one == True]
    x_acc_line_trans = transform_acc(df_line[x_axis], transform)
    y_acc_line_trans = transform_acc(df_line[y_axis], transform)
    lin_fit = scipy.stats.linregress(x_acc_line_trans, y_acc_line_trans)
    slope = lin_fit[0]
    intercept = lin_fit[1]
    lin_fit_ys = xs * slope + intercept
    idx = np.argmax(y_acc_line_trans)
    x_max = x_acc_line_trans[idx]
    y_max = y_acc_line_trans[idx]
    print("ymax",y_max)
    random_slope = y_max/x_max

    print(f'Slope {slope}, intercept {intercept}, r {lin_fit[2]}, pvalue {lin_fit[3]}, stderr {lin_fit[4]}')
    coeffs_ci, fit_lower, fit_upper = get_bootstrap_cis(x_acc_line_trans, y_acc_line_trans, num_bootstrap_samples, xs, 720257663,
                                                        significance_level_coeffs=95, significance_level_grid=95)
    print(f'Bootstrap CIs: {coeffs_ci}')
    sm_model = sm.OLS(y_acc_line_trans, np.stack([np.ones(x_acc_line_trans.shape[0]), x_acc_line_trans], axis=1))
    sm_results = sm_model.fit()
    print(sm_results.summary())

    ax.fill_between(xs, fit_upper, fit_lower, where=xs<pivot_trans, color=f'tab:{fit_color}', alpha=0.3, zorder=6, edgecolor='none', linewidth=0.1)
    fit_line = ax.plot(xs[xs<pivot_trans], lin_fit_ys[xs<pivot_trans], color=f'tab:{fit_color}', zorder=7, linewidth=main_linewidth, alpha=0.5)

    df_line = df[df.use_for_line_fit_two == True]
    x_acc_line_trans = transform_acc(df_line[x_axis], transform)
    y_acc_line_trans = transform_acc(df_line[y_axis], transform)
    lin_fit = scipy.stats.linregress(x_acc_line_trans, y_acc_line_trans)
    slope = lin_fit[0]
    intercept = lin_fit[1]
    lin_fit_ys = xs * slope + intercept
    idx = np.argmax(y_acc_line_trans)
    x_max = x_acc_line_trans[idx]
    y_max = y_acc_line_trans[idx]
    print("ymax",y_max)
    random_slope = y_max/x_max

    print(f'Slope {slope}, intercept {intercept}, r {lin_fit[2]}, pvalue {lin_fit[3]}, stderr {lin_fit[4]}')
    coeffs_ci, fit_lower, fit_upper = get_bootstrap_cis(x_acc_line_trans, y_acc_line_trans, num_bootstrap_samples, xs, 720257663,
                                                        significance_level_coeffs=95, significance_level_grid=95)
    print(f'Bootstrap CIs: {coeffs_ci}')
    sm_model = sm.OLS(y_acc_line_trans, np.stack([np.ones(x_acc_line_trans.shape[0]), x_acc_line_trans], axis=1))
    sm_results = sm_model.fit()
    print(sm_results.summary())

    ax.fill_between(xs, fit_upper, fit_lower, where=xs>=pivot_trans, color=f'tab:{fit_color}', alpha=0.3, zorder=6, edgecolor='none', linewidth=0.1)
    fit_line = ax.plot(xs[xs>pivot_trans], lin_fit_ys[xs>=pivot_trans], color=f'tab:{fit_color}', zorder=7, linewidth=main_linewidth, label='Linear fit (piecewise)', alpha=0.5)

    resnet50 = np.linspace(pivot_trans, pivot_trans, x_plotting_resolution)
    resnet50_line = ax.plot(resnet50, ys, linestyle='dotted', color='black', linewidth=main_linewidth, label='ResNet50 accuracy')

    df_plot = df[df.show_in_plot == True]
    x_acc_plot = df_plot[x_axis]
    y_acc_plot = df_plot[y_axis]
    x_acc_ci = pd.DataFrame(df_plot[f'{x_axis}_ci'].tolist(), index=df_plot.index).to_numpy()
    y_acc_ci = pd.DataFrame(df_plot[f'{y_axis}_ci'].tolist(), index=df_plot.index).to_numpy()
    x_acc_plot_trans = transform_acc(x_acc_plot, transform)
    y_acc_plot_trans = transform_acc(y_acc_plot, transform)
    labels = df_plot.model_type.to_numpy()

    xerr_low = x_acc_plot_trans - transform_acc(x_acc_plot - x_acc_ci[:, 0], transform)
    xerr_high = transform_acc(x_acc_plot + x_acc_ci[:, 1], transform) - x_acc_plot_trans
    xerr = np.stack((xerr_low, xerr_high), axis=0)
    yerr_low = y_acc_plot_trans - transform_acc(y_acc_plot - y_acc_ci[:, 0], transform)
    yerr_high = transform_acc(y_acc_plot + y_acc_ci[:, 1], transform) - y_acc_plot_trans
    yerr = np.stack((yerr_low, yerr_high), axis=0)

    model_points = OrderedDict()
    model_names = []
    for m in model_types:
        if len(m.value) > 2:
            n, c, s = m.value
        else:
            n, c = m.value
            s = defaultmarkersize
        if not any(labels == m):
            continue
        ax.errorbar(x_acc_plot_trans[labels == m], y_acc_plot_trans[labels == m], xerr=xerr[:, labels == m], yerr=yerr[:, labels == m],
                    capsize=2, linewidth=error_line_width, ls='none', color=c, alpha=error_alpha, zorder=8)
        # alpha = None if isinstance(c, (list, tuple)) and len(c) == 4 else 0.5
        points = ax.scatter(x_acc_plot_trans[labels == m], y_acc_plot_trans[labels == m],
                            zorder=9, color=c, s=s, label=n, alpha=alpha, linewidths=0)
        model_points[n] = points
        # Reverse models within groups, then reverse all models. This is the
        # path that gives us the order that matches matplotlib's artists.
        model_names.extend(list(reversed(df_plot[labels == m].index)))
    model_names = model_names[::-1]

    ax.set_xlabel(f'{x_label if x_label else x_axis} ({x_unit})' , fontsize=label_fontsize)
    ax.set_ylabel(f'{y_label if y_label else y_axis} ({y_unit})', fontsize=label_fontsize)
    ax.set_title(f'{title}', fontsize=label_fontsize)
    if set_aspect:
        ax.set_aspect('equal', adjustable='box')
    if include_legend:
        if not include_ideal:
            ax.legend(list(model_points.values()) + [fit_line[0]] + [resnet50_line[0]],
                    list(model_points.keys()) + ['Linear fit (piecewise)'] + ['ResNet50 accuracy'],
                    fontsize=label_fontsize, markerscale=1.5, frameon=False)
        else:
            ax.legend([ideal_repro_line[0]] + list(model_points.values()) + [fit_line[0]] + [resnet50_line[0]],
                    ['y = x'] + list(model_points.keys()) + ['Linear fit (piecewise)'] + ['ResNet50 accuracy'],
                    fontsize=label_fontsize, markerscale=1.5, frameon=False)
    fig.tight_layout()

    if return_separate_legend:  
        fig_legend = matplotlib.pyplot.figure(figsize=(25, 2)) 
        fig_legend.legend([ideal_repro_line[0]] + list(model_points.values()) + [fit_line[0]] + [resnet50_line[0]], 
                          ['y = x'] + list(model_points.keys()) + ['Linear fit (piecewise)'] + ['ResNet50 accuracy'],
                          fontsize=label_fontsize, ncol=num_legend_columns, markerscale=1.5, 
                          loc='center', frameon=False) 
        fig_legend.tight_layout()    
        return fig, ax, fig_legend  

    elif return_ordered_names:
        return fig, ax, model_names

    else:
        return fig, ax


def model_scatter_plot_interactive(df, x_axis, y_axis, xlim, ylim, model_types, num_bootstrap_samples,
                       title, transform='logit', x_unit='top-1, %', y_unit='top-1, %', include_legend=True,
                       tick_multiplier=10, extra_x_ticks=[], extra_y_ticks=[], set_aspect=False,
                       x_label=None, y_label=None, height=700, width=700, 
                       include_ideal=True, fit_color="red", bl_value=None, 
                       bl_name=None, return_separate_legend=False, num_legend_columns=3, error_line_width=0.5, error_alpha=0.5,
                       alpha=0.5, x_tick_multiplier=None, y_tick_multiplier=None):

    assert (df[df.show_in_plot == True][x_axis] <= xlim[1]).all()
    assert (df[df.show_in_plot == True][x_axis] >= xlim[0]).all()
    assert (df[df.show_in_plot == True][y_axis] <= ylim[1]).all()
    assert (df[df.show_in_plot == True][y_axis] >= ylim[0]).all()
    print('ylim range / xlim range aspect ratio: ', (ylim[1] - ylim[0]) / (xlim[1] - xlim[0]))

    x_tick_multiplier = x_tick_multiplier if x_tick_multiplier is not None else tick_multiplier
    y_tick_multiplier = y_tick_multiplier if y_tick_multiplier is not None else tick_multiplier
    tick_loc_x = np.array(sorted(tick_locs(xlim[0], xlim[1], x_tick_multiplier) + extra_x_ticks))
    tick_text_x = [str(int(loc)) for loc in tick_loc_x]
    tick_loc_y = np.array(sorted(tick_locs(ylim[0], ylim[1], y_tick_multiplier) + extra_y_ticks))
    tick_text_y = [str(int(loc)) for loc in tick_loc_y]

    df_line = df[df.use_for_line_fit == True]
    x_acc_line_trans = transform_acc(df_line[x_axis], transform)
    y_acc_line_trans = transform_acc(df_line[y_axis], transform)

    xs = np.linspace(transform_acc(xlim[0], transform), transform_acc(xlim[1], transform), x_plotting_resolution)
    lin_fit = scipy.stats.linregress(x_acc_line_trans, y_acc_line_trans)
    slope = lin_fit[0]
    intercept = lin_fit[1]
    lin_fit_ys = xs * slope + intercept
    idx = np.argmax(y_acc_line_trans)
    x_max = x_acc_line_trans[idx]
    y_max = y_acc_line_trans[idx]
    print("ymax",y_max)
    random_slope = y_max/x_max

    print(f'Slope {slope}, intercept {intercept}, r {lin_fit[2]}, pvalue {lin_fit[3]}, stderr {lin_fit[4]}')
    coeffs_ci, fit_lower, fit_upper = get_bootstrap_cis(x_acc_line_trans, y_acc_line_trans, num_bootstrap_samples, xs, 720257663,
                                                        significance_level_coeffs=95, significance_level_grid=95)
    print(f'Bootstrap CIs: {coeffs_ci}')
    sm_model = sm.OLS(y_acc_line_trans, np.stack([np.ones(x_acc_line_trans.shape[0]), x_acc_line_trans], axis=1))
    sm_results = sm_model.fit()
    print(sm_results.summary())

    df_plot = df[df.show_in_plot == True].copy()
    x_acc_plot = df_plot[x_axis]
    y_acc_plot = df_plot[y_axis]
    x_acc_ci = pd.DataFrame(df_plot[f'{x_axis}_ci'].tolist(), index=df_plot.index).to_numpy()
    y_acc_ci = pd.DataFrame(df_plot[f'{y_axis}_ci'].tolist(), index=df_plot.index).to_numpy()
    x_acc_plot_trans = transform_acc(x_acc_plot, transform)
    y_acc_plot_trans = transform_acc(y_acc_plot, transform)
    labels = df_plot.model_type.to_numpy()

    df_plot['x_acc_plot_trans'] = x_acc_plot_trans
    df_plot['y_acc_plot_trans'] = y_acc_plot_trans
    df_plot['model_type_name'] = [x.value[0] for x in df_plot.model_type]

    fig = px.scatter(df_plot, x='x_acc_plot_trans', y='y_acc_plot_trans',
                    hover_name=df_plot.index, 
                    hover_data={x_axis: ':.3f', y_axis: ':.3f', 'x_acc_plot_trans': False, 'y_acc_plot_trans': False, 'model_type_name': True},
                    color=[x.value[1].replace('tab:', '') for x in labels],
                    color_discrete_map={'green': 'green', 'blue': 'blue', 'brown': 'brown', 'olive': 'orange'},
                    height=height, width=width)

    fig.update_layout(
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        title=go.layout.Title(
            text=title,
            font=dict(
                size=15
            ),
            x=0.5
        ),
        xaxis=go.layout.XAxis(
            tickmode='array',
            tickvals=transform_acc(tick_loc_x, transform),
            ticktext=tick_text_x,
            title=go.layout.xaxis.Title(
                text=f'{x_label if x_label else x_axis} ({x_unit})',
                font=dict(
                    size=15,
                )
            )
        ),
        yaxis=go.layout.YAxis(
            tickmode='array',
            tickvals=transform_acc(tick_loc_y, transform),
            ticktext=tick_text_y,
            title=go.layout.yaxis.Title(
                text=f'{y_label if y_label else y_axis} ({y_unit})',
                font=dict(
                    size=15,
                )
            )
        ),
        shapes=[
            # Diagonal line
            go.layout.Shape(
                type="line",
                x0=min(transform_acc(xlim, transform)[0], transform_acc(ylim, transform)[0]),
                y0=min(transform_acc(xlim, transform)[0], transform_acc(ylim, transform)[0]),
                x1=max(transform_acc(xlim, transform)[1], transform_acc(ylim, transform)[1]),
                y1=max(transform_acc(xlim, transform)[1], transform_acc(ylim, transform)[1]),
                line=dict(
                    color="Black",
                    width=2,
                    dash="dash",
                ),
            ),
            # Least squares fit line
            go.layout.Shape(
                type="line",
                x0=transform_acc(xlim, transform)[0],
                y0=transform_acc(xlim, transform)[0] * slope + intercept,
                x1=transform_acc(xlim, transform)[1],
                y1=transform_acc(xlim, transform)[1] * slope + intercept,
                line=dict(
                    color="Red",
                    width=2,
                ),
            ),
        ]
    )

    fig.update_xaxes(range=transform_acc(xlim, transform))
    fig.update_yaxes(range=transform_acc(ylim, transform))
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey',
                     zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey',
                     showline=True, linewidth=1, linecolor='Black', mirror=True)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey',
                     zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey',
                     showline=True, linewidth=1, linecolor='Black', mirror=True)

    return fig
