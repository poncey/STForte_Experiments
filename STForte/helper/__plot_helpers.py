from __future__ import annotations
import pymde
import numpy as np
import pandas as pd
import matplotlib as mpl
import plotly.express as px
import matplotlib.pyplot as plt
import scipy.stats as stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.sparse import csr_matrix
from anndata._core.anndata import AnnData
from typing import Union, List, Dict
from IPython.display import Image
from matplotlib.colors import LinearSegmentedColormap
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.plotting import figure
from bokeh.transform import factor_cmap


def mde(adata: AnnData, 
        use_rep: str = None, 
        preserving: str = "neighbors",
        verbose: bool = False,
        device: str = "cuda",
        mde_kwargs={},
        field_name: str | None = None, 
        copy: bool = False
        ):
    """_summary_

    Args:
        adata (AnnData): Input AnnData object for scRNA/ST data.
        use_rep (str, optional): Use the indicated representation referenced in `.obsm`. If None, adata.X is used. Defaults to None.
        preserving (str, optional): Specify the method (whether to preserve neighbors or distances information) of pymde. Defaults to "neighbours".
        verbose (bool, optional): Whether to print pymde verbose. Defaults to False.
        device (str, optional): Specify the device to perform pymde. Defaults to "cuda".
        mde_kwargs (dict, optional): Other arguments for performing pymde. Defaults to {}.
        field_name (str, optional): field name for pymde when return results to AnnData (copy is False). 
            If None, the field_name is specified as: "{use_rep}_MDE". Defaults to None.
        copy (bool, optional): Choose to return the MDE results directly or append to the input `AnnData.obsm`. Defaults to False.

    Returns:
        embedd (np.nd_array): Only return the MDE outputs when `copy=True`.
    """
    if use_rep is None:
        if isinstance(adata.X, csr_matrix):
            X = adata.X.todense().A
        else:
            X = adata.X
        use_rep = "X"
    else:
        X = adata.obsm[use_rep]
    
    if preserving == "neighbors":
        model = pymde.preserve_neighbors(X, verbose=verbose, device=device, **mde_kwargs)
    elif preserving == "distances":
        model = pymde.preserve_distances(X, verbose=verbose, device=device, **mde_kwargs)
    else:
        raise ValueError("the preserving parameter is whether `neighbors` or `distances`.")
    embedd = model.embed(verbose=verbose).cpu().detach().numpy()
    
    if copy:
        return embedd
    else:
        field_name = "{:s}_MDE".format(use_rep) if field_name is None else field_name
        adata.obsm[field_name] = embedd


def convert_category_colors(cateseries,
                           with_catenames=False,
                           color_palette: list = None):
    catenames = cateseries.cat.categories
    catelen = catenames.__len__()
    if color_palette is None:
        if catelen <= 10:
            cate_color = cateseries.cat.rename_categories(
                px.colors.qualitative.Prism[:catelen])
        elif catelen <= 26:
            cate_color = cateseries.cat.rename_categories(
                iwanthue_alphabet_hard[:catelen])
        elif catelen <= 42:
            cate_color = cateseries.cat.rename_categories(
                iwanthue_answer_hard[:catelen])
        elif catelen <= 102:
            cate_color = cateseries.cat.rename_categories(
                godsnot_102[:catelen])
        else:
            raise ValueError(
                "Too much categories for discrete color palettes.")
    else:
        cate_color = cateseries.cat.rename_categories(color_palette[:catelen])
    if with_catenames:
        return cate_color, catenames
    else:
        return cate_color


def plot_embeddings(adata, basis, color, size=4, 
                    title=None,
                    width=640, height=640, 
                    palette=px.colors.qualitative.Plotly,
                    return_source=True):
    s = ColumnDataSource({
        'x': adata.obsm[basis][:, 0],
        'y': adata.obsm[basis][:, 1],
        color: adata.obs[color],
    })
    p = figure(title=title, width=width, height=height, toolbar_location=None)
    p.circle(
        'x', 'y', source=s,
        fill_color=factor_cmap(color, palette=palette, factors=adata.obs[color].cat.categories),
        size=size, line_color=None,)
    if return_source:
        return p, s
    else:
        return p
    
    
def layout_centroid_label(s, label, x= 'x', y='y',
                          font_size=32,):
    categories = s.data[label].cat.categories
    centroid_x = [s.data['x'][s.data[label] == cc].mean() for cc in categories]
    centroid_y = [s.data['y'][s.data[label] == cc].mean() for cc in categories]
    centroid_source = ColumnDataSource(dict(x=centroid_x, y=centroid_y, annotation=categories))
    labels = LabelSet(
        x=x, y=y, text='annotation', source=centroid_source,
        text_outline_color="#0e0e0e", text_color="#fafafa",
        text_font="Arial", text_font_size=f'{font_size}pt', text_font_style = 'bold',
                    )
    return labels


def plot_foi_overlay(adata,
                     coor_loc: Union[str, List[str]] = ['obsm', 'spatial'],
                     foi: Union[str, np.array, list] = None,
                     foi_alias: Union[str, list] = None,
                     title=None,
                     marker_size: float = 1.,
                     layout_params: dict = {},
                     opacity: float = 1,
                     continuous_color_palette=None,
                     discrete_color_palette=None,
                     disable_axes: bool = True,
                     scatterkwargs: dict = {}):
    if isinstance(foi, list):
        # TODO: Multiple foi subplots.
        fig = make_subplots()
    else:
        fig = go.Figure()

    if marker_size is None:
        marker_size = 1

    if opacity is not None:
        opacity = np.clip(opacity, a_min=0, a_max=1)

    if foi is not None:
        if isinstance(foi, str):
            foi = adata.obs[foi]
    else:
        # plot default ST foi.
        foi = ["blue"]*adata.X.shape[0]
        foi_alias = "spot"
        
    coor = slice_stdata(adata, coor_loc)

    # manipulate categorical foi
    if isinstance(foi, pd.Series) and (foi.dtype.name == "category"):
        foi_color, catenames = convert_category_colors(foi, with_catenames=True, color_palette=discrete_color_palette)
        for cn in catenames:
            catind = (foi == cn)
            fig.add_trace(go.Scatter(x=coor[:, 0][catind],
                                        y=coor[:, 1][catind], mode='markers',
                                        marker=dict(color=foi_color[catind],
                                                    size=marker_size, opacity=opacity),
                                        name=cn, **scatterkwargs))
            fig.update_layout(showlegend=True, legend_title_text=foi_alias)
    # manipulate contiunous foi
    else:
        fig.add_trace(go.Scatter(x=coor[:, 0], y=coor[:, 1], mode='markers',
                                 marker=dict(color=foi, size=marker_size, opacity=opacity,
                                             colorscale=continuous_color_palette,
                                             colorbar=dict(title=""),),
                                 name=foi_alias, **scatterkwargs))
        # fig.update_layout(coloraxis_colorbar=dict(title=foi_alias))

    if disable_axes:
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
    if 'template' not in layout_params:
        layout_params = layout_params.copy()
        layout_params['template'] = 'plotly_white'
    fig.update_layout(title=title, **layout_params)
    return fig


def plot_embeddings_plotly(adata, prop, embed,
                           prop_alias="",
                           embed_alias="",
                           title=None,
                           marker_size=None,
                           opacity=None,
                           discrete_color_palette=None,
                           continuous_color_palette=None):
    if isinstance(prop, str):
        prop = adata.obs[prop]
    if isinstance(embed, str):
        embed = adata.obsm[embed]

    # manipulate categorical embeddings
    if isinstance(prop, pd.Series) and (prop.dtype.name == "category"):
        fig = go.Figure()
        prop_color, catenames = convert_category_colors(
            prop, with_catenames=True, color_palette=discrete_color_palette)
        for cn in catenames:
            catind = (prop == cn)
            fig.add_trace(go.Scatter(x=embed[:, 0][catind], y=embed[:, 1][catind],
                                     mode='markers',
                                     marker=dict(color=prop_color[catind],
                                                 size=marker_size, opacity=opacity),
                                     name=cn))
        fig.update_layout(showlegend=True, legend_title_text=prop_alias)
    else:
        fig = px.scatter(x=embed[:, 0],
                         y=embed[:, 1], color=prop,
                         colorscale=continuous_color_palette)
        fig.update_layout(coloraxis_colorbar=dict(title=prop_alias))
    fig.update_layout(legend_title_text=prop_alias,
                      xaxis_title="{:s} axis-1".format(embed_alias),
                      yaxis_title="{:s} axis-2".format(embed_alias),
                      title=title)
    fig.update_layout(legend=dict(itemsizing='constant',
                                orientation='h',
                                title_side='top',
                                x=0, y=-0.15
                                ),
                    margin=dict(l=20, r=20, b=20, t=20, pad=10),
                    template="plotly_white")
    fig.update_yaxes(title_standoff=0, showticklabels=False)
    fig.update_xaxes(title_standoff=0, showticklabels=False)
    return fig


def show_static_figure(fig, **kwargs):
    img_bytes = fig.to_image(format="png", **kwargs)
    img = Image(img_bytes)
    return img


def slice_stdata(
        stdata:AnnData,
        slice: List[str]
        ):
        assert slice[0] in ['obs','var','obsm','uns']

        if slice[0] in ['obs','var']:
            tar = stdata.obs if slice[0]=='obs' else stdata.var
            if len(slice) == 2:
                return np.array(tar[slice[1]])
            elif len(slice) > 2:
                if isinstance(tar,pd.DataFrame):
                    return np.array(tar[[*slice[1:]]])
                elif isinstance(tar, Dict):
                    return np.concatenate(
                        [np.array(tar[slice[i+1]]) for i in range(len(slice)-1)],
                        axis=1
                    )
            else:
                raise ValueError('Invalide keys for obs/var extraction')
        elif slice[0] == 'obsm':
            return np.array(stdata.obsm[slice[1]])
        elif slice[0] == 'uns':
            tar=stdata.uns
            for key in slice[1:]:
                tar = tar[key]
            return np.array(tar)


def plot_trend_genes(
    adata: AnnData,
    genelist: Union[list, pd.Index],
    group: Union[str, pd.Series],
    title = None,
    line_color: str = '#F8F8FF',
    line_width: float = 1.5,
    marker_palette: Union[list, dict] = px.colors.qualitative.Plotly,
    ):
    if isinstance(group, str):
        group = adata.obs[group]
    cat_ind = group.cat.categories
    fig = make_subplots(rows=len(genelist),)
    # plot trend lines
    for ii, gg in enumerate(genelist):
        try:
            valmean = [adata[group == cc][:, gg].X.A.squeeze().mean() for cc in cat_ind]
        except AttributeError:
            valmean = [adata[group == cc][:, gg].X.squeeze().mean() for cc in cat_ind]
        c = [cc for cc in cat_ind]
        try:
            valstd = [adata[group == cc][:, gg].X.A.squeeze().std() for cc in cat_ind]
        except AttributeError:
            valstd = [adata[group == cc][:, gg].X.squeeze().std() for cc in cat_ind]
        fig.add_trace(
            go.Scatter(
                y=valmean,
                x=c,
                line=dict(color=line_color, width=line_width),
            ),
            row=ii+1, col=1,
        )
        # plot error bars
        for jj, cc in enumerate(cat_ind):
            c_marker = marker_palette[cc] if isinstance(marker_palette, dict) else marker_palette[jj]
            fig.add_trace(go.Scatter(
                        x=[cc],
                        y=[valmean[jj]],
                        mode='markers+text',
                        textposition='top center',
                        error_y=dict(
                            type='data',
                            color=c_marker,
                            array=[valstd[jj]],
                            visible=True),
                        marker=dict(color=c_marker),
                        showlegend=False
                    ), row=ii+1, col=1,)
    for ii in range(len(genelist)):
        if ii != len(genelist) - 1:
            fig.update_xaxes(showticklabels=False, row=ii+1, col=1)
        fig.update_yaxes(title=genelist[ii], title_standoff=0, row=ii+1, col=1)
    # fig.update_xaxes(side='top')
    fig.update_yaxes(showticklabels=False, title_font_size=12)
    fig.update_layout(title=title,
                      margin=dict(l=16, r=16, b=16, t=20, pad=10),
                      showlegend=False,)
    return fig


def add_p_value_annotation(fig, array_columns, subplot=None, _format=dict(interline=0.07, text_height=1.07, color='darkslategray'),
                           show_pval=True,):
    ''' Adds notations giving the p-value between two box plot data (t-test two-sided comparison)
    
    Parameters:
    ----------
    fig: figure
        plotly boxplot figure
    array_columns: np.array
        array of which columns to compare 
        e.g.: [[0,1], [1,2]] compares column 0 with 1 and 1 with 2
    subplot: None or int
        specifies if the figures has subplots and what subplot to add the notation to
    _format: dict
        format characteristics for the lines

    Returns:
    -------
    fig: figure
        figure with the added notation
    '''
    # Specify in what y_range to plot for each pair of columns
    y_range = np.zeros([len(array_columns), 2])
    for i in range(len(array_columns)):
        y_range[i] = [1.01+i*_format['interline'], 1.02+i*_format['interline']]

    # Get values from figure
    fig_dict = fig.to_dict()

    # Get indices if working with subplots
    if subplot:
        if subplot == 1:
            subplot_str = ''
        else:
            subplot_str =str(subplot)
        indices = [] #Change the box index to the indices of the data for that subplot
        for index, data in enumerate(fig_dict['data']):
            #print(index, data['xaxis'], 'x' + subplot_str)
            if data['xaxis'] == 'x' + subplot_str:
                indices = np.append(indices, index)
        indices = [int(i) for i in indices]
        # print((indices))
    else:
        subplot_str = ''

    # Print the p-values
    for index, column_pair in enumerate(array_columns):
        if subplot:
            data_pair = [indices[column_pair[0]], indices[column_pair[1]]]
        else:
            data_pair = column_pair

        # Mare sure it is selecting the data and subplot you want
        #print('0:', fig_dict['data'][data_pair[0]]['name'], fig_dict['data'][data_pair[0]]['xaxis'])
        #print('1:', fig_dict['data'][data_pair[1]]['name'], fig_dict['data'][data_pair[1]]['xaxis'])

        # Get the p-value
        pvalue = stats.ttest_ind(
            fig_dict['data'][data_pair[0]]['y'],
            fig_dict['data'][data_pair[1]]['y'],
            equal_var=False,
        )[1]
        if show_pval:
            symbol = f"P = {pvalue:.3f}"
        else:
            if pvalue >= 0.05:
                symbol = 'ns'
            elif pvalue >= 0.01: 
                symbol = '*'
            elif pvalue >= 0.001:
                symbol = '**'
            else:
                symbol = '***'
        # Vertical line
        fig.add_shape(type="line",
            xref="x"+subplot_str, yref="y"+subplot_str+" domain",
            x0=column_pair[0], y0=y_range[index][0], 
            x1=column_pair[0], y1=y_range[index][1],
            line=dict(color=_format['color'], width=2,)
        )
        # Horizontal line
        fig.add_shape(type="line",
            xref="x"+subplot_str, yref="y"+subplot_str+" domain",
            x0=column_pair[0], y0=y_range[index][1], 
            x1=column_pair[1], y1=y_range[index][1],
            line=dict(color=_format['color'], width=2,)
        )
        # Vertical line
        fig.add_shape(type="line",
            xref="x"+subplot_str, yref="y"+subplot_str+" domain",
            x0=column_pair[1], y0=y_range[index][0], 
            x1=column_pair[1], y1=y_range[index][1],
            line=dict(color=_format['color'], width=2,)
        )
        ## add text at the correct x, y coordinates
        ## for bars, there is a direct mapping from the bar number to 0, 1, 2...
        fig.add_annotation(dict(font=dict(color=_format['color'],size=16),
            x=(column_pair[0] + column_pair[1])/2,
            y=y_range[index][1]*_format['text_height'],
            showarrow=False,
            text=symbol,
            textangle=0,
            xref="x"+subplot_str,
            yref="y"+subplot_str+" domain"
        ))
    return fig


def compare_unseen_expression(adata:AnnData,
                              adata_sp:AnnData,
                              gene_name: str,
                              indexer_ori: Union[str, pd.Series] = "spot_instance",
                              ):
    fig = make_subplots(rows=1, cols=3, column_widths=[0.4, 0.4, 0.2],
                        row_heights=[0.4],
                        subplot_titles=("Source", "Padded", ""))
    fig.add_trace(
        go.Scatter(
            x=adata.obsm['spatial'][:, 0],
            y=-adata.obsm['spatial'][:, 1],
            marker_color=adata[:, f'{gene_name}'].X.A.squeeze(),
            marker_size=6.0,
            mode="markers",
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=adata_sp.obsm['spatial'][:, 0],
            y=-adata_sp.obsm['spatial'][:, 1],
            marker_color=adata_sp.obs[f'{gene_name}_with_padding'],
            mode="markers",
            marker_size=3.0,
            marker_colorbar=dict(
                thicknessmode="pixels", thickness=24,
                lenmode="pixels", len=320,
                orientation='h',
                xanchor="left", x=0.2,
                yanchor="top", y=0.05,
                tickvals=[adata_sp.obs[f'{gene_name}_with_padding'].min(), adata_sp.obs[f'{gene_name}_with_padding'].max()],
                ticktext=["low", "high"],
                tickfont=dict(size=24),
                )
        ),
        row=1, col=2
    )
    fig.update_layout(width=1000, height=500, 
                      minreducedwidth=500,
                      minreducedheight=500,
                      showlegend=False, template="plotly_white")
    for ii in [1, 2]:
        fig.update_xaxes(visible=False, row=1, col=ii)
        fig.update_yaxes(visible=False, row=1, col=ii)
    y0 = adata_sp[adata_sp.obs[indexer_ori] == 'Observed'].obs[f'{gene_name}_with_padding']
    y1 = adata_sp[adata_sp.obs[indexer_ori] == 'Inferred'].obs[f'{gene_name}_with_padding']
    y3 = adata_sp.obs[f'{gene_name}_with_padding']
    fig.add_trace(go.Violin(y=y0, name='Observed',
                            line_color=iwanthue_alphabet_hard[1],
                            spanmode = 'hard'),
                row=1, col=3)
    fig.add_trace(go.Violin(y=y1, name='Inferred',
                            line_color=iwanthue_alphabet_hard[0],
                            spanmode = 'hard'),
                row=1, col=3)
    fig.add_trace(go.Violin(y=y3, name='All',
                            line_color="#808880",
                            spanmode = 'hard'),
                row=1, col=3)
    fig.update_traces(line_width=2., meanline_visible=True, row=1, col=3)
    fig.update_xaxes(tickangle=30, tickson="boundaries", row=1, col=3)
    fig.update_layout(violingap=0, violinmode='overlay')
    fig.update_layout(font_size=22, font_family="Arial",
                      title=dict(text=gene_name, font=dict(size=32), yref='paper', x=0.1),)
    fig.update_traces(
        marker=dict(symbol="hexagon", colorscale=px.colors.sequential.Peach),
        selector=dict(mode="markers"), 
    )
    fig.update_annotations(font_size=26)
    return fig


def plot_foi_genes(
        adata,
        genelist: list,
        title=None,
        marker_size: float = 2.2,
        facet_cols: int = 8,
        color_continuous_pallete='inferno',
        template='plotly_dark',
        normalize_colorscale=True,
        **kwargs):

    # Manufacture DataFrame for plotly.express visualization
    dfgl = []
    for gn in genelist:
        dfg = pd.DataFrame(adata[:, gn].obsm['spatial'],
                           columns=['x', 'y'],)

        # minmax-scaling to uniform colorscale
        valg = adata[:, gn].X.A.flatten()
        if normalize_colorscale:
            valg = (valg - valg.min()) / (valg.max() - valg.min())
        dfg.insert(0, 'value', valg)
        dfg.insert(0, 'gene', [gn] * adata.shape[0])
        dfgl.append(dfg)
    dfgl = pd.concat(dfgl)

    # Plot the spatial distribution for genes
    fig = px.scatter(dfgl, x='x', y='y', facet_col='gene', color='value',
                     facet_col_wrap=facet_cols,
                     color_continuous_scale=color_continuous_pallete,
                     template=template, **kwargs)
    fig.update_traces(marker_size=marker_size)
    fig.update_coloraxes(showscale=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    if title is not None:
        fig.update_layout(title=title)
    return fig


def rgb2hex(rgb_codes):
    hex_codes = []
    for rgb_code in rgb_codes:
        # Extracting RGB values
        rgb_values = rgb_code[rgb_code.index('(') + 1: rgb_code.index(')')].split(',')
        red, green, blue = map(int, rgb_values)

        # Converting RGB to hexadecimal
        hex_code = f'#{red:02X}{green:02X}{blue:02X}'
        hex_codes.append(hex_code)
    return hex_codes


def hex2rgb(hex_codes: Union[str, list], csslike: bool = True):
    if not isinstance(hex_codes, list):
        hex_codes = [hex_codes]
    rgbcodes = []
    for hex_code in hex_codes:
        hex_code = hex_code.lstrip('#')  # Remove '#' if present
        if len(hex_code) == 3:  # Convert short HEX code to full length (e.g., #ABC to #AABBCC)
            hex_code = ''.join(c * 2 for c in hex_code)
        
        r = int(hex_code[0:2], 16)
        g = int(hex_code[2:4], 16)
        b = int(hex_code[4:6], 16)
        if csslike:
            rgbcodes.append(f'rgb({r}, {g}, {b})') 
        else:
            rgbcodes.append((r, g, b))
    return rgbcodes


def colormap_with_alpha(cmap:mpl.colors.LinearSegmentedColormap,
                        alpha_scale: float = 20):
    a_cmap = cmap(np.arange(cmap.N))
    xx = np.linspace(0, 1, cmap.N)
    yy = -np.exp(-alpha_scale*xx) + 1
    a_cmap[:, -1] = yy
    a_cmap = mpl.colors.ListedColormap(a_cmap)
    return a_cmap


def rgb_string_to_tuple(rgb_string):
    # Remove 'rgb(' and ')' from the string
    rgb_string = rgb_string.replace('rgb(', '').replace(')', '')
    
    # Split the string into individual components
    r, g, b = rgb_string.split(',')
    
    # Convert the components to integers and create a tuple
    rgb_tuple = (int(r), int(g), int(b))
    
    return rgb_tuple


def rgb_string_to_normalized_tuple(rgb_string):
    # Remove 'rgb(' and ')' from the string
    rgb_string = rgb_string.replace('rgb(', '').replace(')', '')
    
    # Split the string into individual components
    r, g, b = rgb_string.split(',')
    
    # Convert the components to integers and create a tuple
    r_normalized = int(r) / 255.0
    g_normalized = int(g) / 255.0
    b_normalized = int(b) / 255.0
    
    normalized_rgb_tuple = (r_normalized, g_normalized, b_normalized)
    
    return normalized_rgb_tuple


def create_refined_colormap(color_palette, stages:int = 256):
    colors_rgb = [rgb_string_to_normalized_tuple(color) for color in color_palette]
    positions = np.linspace(0, 1, len(colors_rgb))

    refined_positions = np.linspace(0, 1, stages)
    refined_colormap = []

    for i in range(3):  # Three channels: red, green, blue
        channel_colors = [color[i] for color in colors_rgb]
        interp_channel = np.interp(refined_positions, positions, channel_colors)
        refined_colormap.append(interp_channel)

    refined_colormap = np.transpose(refined_colormap)
    refined_colormap = tuple(map(tuple, refined_colormap))

    return LinearSegmentedColormap.from_list('RefinedColormap', refined_colormap)


# Color palettes
# from http://godsnotwheregodsnot.blogspot.de/2012/09/color-distribution-methodology.html
godsnot_102 = [
    # "#000000",  # remove the black, as often, we have black colored annotation
    "#FFFF00",
    "#1CE6FF",
    "#FF34FF",
    "#FF4A46",
    "#008941",
    "#006FA6",
    "#A30059",
    "#FFDBE5",
    "#7A4900",
    "#0000A6",
    "#63FFAC",
    "#B79762",
    "#004D43",
    "#8FB0FF",
    "#997D87",
    "#5A0007",
    "#809693",
    "#6A3A4C",
    "#1B4400",
    "#4FC601",
    "#3B5DFF",
    "#4A3B53",
    "#FF2F80",
    "#61615A",
    "#BA0900",
    "#6B7900",
    "#00C2A0",
    "#FFAA92",
    "#FF90C9",
    "#B903AA",
    "#D16100",
    "#DDEFFF",
    "#000035",
    "#7B4F4B",
    "#A1C299",
    "#300018",
    "#0AA6D8",
    "#013349",
    "#00846F",
    "#372101",
    "#FFB500",
    "#C2FFED",
    "#A079BF",
    "#CC0744",
    "#C0B9B2",
    "#C2FF99",
    "#001E09",
    "#00489C",
    "#6F0062",
    "#0CBD66",
    "#EEC3FF",
    "#456D75",
    "#B77B68",
    "#7A87A1",
    "#788D66",
    "#885578",
    "#FAD09F",
    "#FF8A9A",
    "#D157A0",
    "#BEC459",
    "#456648",
    "#0086ED",
    "#886F4C",
    "#34362D",
    "#B4A8BD",
    "#00A6AA",
    "#452C2C",
    "#636375",
    "#A3C8C9",
    "#FF913F",
    "#938A81",
    "#575329",
    "#00FECF",
    "#B05B6F",
    "#8CD0FF",
    "#3B9700",
    "#04F757",
    "#C8A1A1",
    "#1E6E00",
    "#7900D7",
    "#A77500",
    "#6367A9",
    "#A05837",
    "#6B002C",
    "#772600",
    "#D790FF",
    "#9B9700",
    "#549E79",
    "#FFF69F",
    "#201625",
    "#72418F",
    "#BC23FF",
    "#99ADC0",
    "#3A2465",
    "#922329",
    "#5B4534",
    "#FDE8DC",
    "#404E55",
    "#0089A3",
    "#CB7E98",
    "#A4E804",
    "#324E72",
]   


# Alphabet colors generated from https://medialab.github.io/iwanthue/
iwanthue_alphabet_hard = [
    "#009480",
    "#fd325b",
    "#3ee26f",
    "#364fd4",
    "#8bd840",
    "#e387ff",
    "#bfd034",
    "#bf007f",
    "#00a656",
    "#ac0036",
    "#00c9a3",
    "#9f2600",
    "#699cff",
    "#cca400",
    "#aeaaff",
    "#e16d00",
    "#005d91",
    "#fdb879",
    "#8f2a6b",
    "#0a6300",
    "#ff9dcd",
    "#606000",
    "#71d3f4",
    "#ffa289",
    "#a15c65",
    "#cb8285",
]

# 42 colors generated from https://medialab.github.io/iwanthue/
iwanthue_answer_hard = [
    "#eb9300",
    "#9c61eb",
    "#189c0f",
    "#c857dd",
    "#80db6b",
    "#6436b1",
    "#f8bc2a",
    "#015ec2",
    "#c9cd4e",
    "#c60098",
    "#00b36c",
    "#f23bb0",
    "#017428",
    "#eb0067",
    "#00865d",
    "#ff529f",
    "#376200",
    "#8889ff",
    "#ac9700",
    "#0285d3",
    "#da6b00",
    "#00c9f1",
    "#db4615",
    "#01adde",
    "#f14536",
    "#6cc8ff",
    "#a15a00",
    "#b3bfff",
    "#a31432",
    "#93d5a3",
    "#a00d5c",
    "#008d7d",
    "#ff78c6",
    "#c3cc88",
    "#882d7a",
    "#ff9a67",
    "#73a3d4",
    "#7f4327",
    "#ff9bba",
    "#7a4366",
    "#d5a77d",
    "#8e3248",
]

# 102 colors generated from https://medialab.github.io/iwanthue/
iwanthue_102_hard = [
    "#6c3dc2",
    "#a1cf27",
    "#3e4bcf",
    "#43a100",
    "#9738bf",
    "#18d666",
    "#b5009b",
    "#36af29",
    "#9360ea",
    "#8cda51",
    "#1261e8",
    "#ecc22c",
    "#6376ff",
    "#c8b000",
    "#7184ff",
    "#4c9500",
    "#eb7bff",
    "#008812",
    "#e65cdd",
    "#007500",
    "#e23cb9",
    "#009c43",
    "#f52793",
    "#66dd8c",
    "#d4007b",
    "#00a869",
    "#f10d6e",
    "#02ba96",
    "#d10049",
    "#38dade",
    "#b31800",
    "#01c2dc",
    "#ff5f36",
    "#026ad0",
    "#e8a200",
    "#5d3daa",
    "#dcc740",
    "#ad81ff",
    "#638600",
    "#dd90ff",
    "#9b9500",
    "#911b88",
    "#c2ce67",
    "#0158ae",
    "#fb8918",
    "#56a4ff",
    "#ff742f",
    "#0282b6",
    "#c96600",
    "#6fd2f8",
    "#b20018",
    "#8cd5b0",
    "#cc0062",
    "#00722e",
    "#ff67c0",
    "#4c6800",
    "#be9fff",
    "#a28b00",
    "#cdafff",
    "#c37700",
    "#b9baff",
    "#aa7300",
    "#f8acfd",
    "#3e5b0e",
    "#ff70b9",
    "#01754c",
    "#ff4e61",
    "#0f5e44",
    "#ff547a",
    "#bccf77",
    "#84307d",
    "#d3c86e",
    "#4a4d87",
    "#e5c367",
    "#773f6b",
    "#fdb96b",
    "#9595cb",
    "#ff6147",
    "#3f5a26",
    "#ff77a9",
    "#686b00",
    "#c40030",
    "#56551e",
    "#ff8087",
    "#685001",
    "#c88fb7",
    "#895e00",
    "#88345f",
    "#f0bc88",
    "#932e43",
    "#ffb2a2",
    "#9e231a",
    "#eaa791",
    "#9a3a00",
    "#e398a0",
    "#933130",
    "#a48355",
    "#ff9370",
    "#9a565a",
    "#ff9f91",
    "#865838",
    "#a36854",
]

# 32 colors generated by kmeans
iwanthue_32_soft = [
    "#4baebc",
    "#e0462a",
    "#593ccf",
    "#60bf37",
    "#a83bd7",
    "#4eba68",
    "#d851ce",
    "#aeb538",
    "#4d2992",
    "#5d7e2c",
    "#816be1",
    "#d67a31",
    "#5077cd",
    "#cc9e4e",
    "#8f429d",
    "#5bb18e",
    "#db4092",
    "#36613e",
    "#d73a5b",
    "#77a0d0",
    "#973727",
    "#b48bd0",
    "#484218",
    "#d679af",
    "#a0ae72",
    "#46346f",
    "#8d6337",
    "#585f85",
    "#db7670",
    "#8e3265",
    "#cb9298",
    "#6c2f39"
]

# Prism Light palette from GraphPad Prism
prism_light_palette = [
    "#A48AD3",
    "#1CC5FE",
    "#6FC7CF",
    "#FBA27D",
    "#FB7D80",
    "#2C1453",
    "#114CE8",
    "#0E6F7C",
    "#FB4F06",
    "#FB0005",
]

# Prism Dark palette from GraphPad Prism
prism_dark_palette = [
    "#2C1453",
    "#114CE8",
    "#0E6F7C",
    "#FB4F06",
    "#FB0005",
    "#A48AD3",
    "#1CC5FE",
    "#6FC7CF",
    "#FBA27D",
    "#FB7D80",
]

# 1960s palette from GraphPad Prism
prism_1960s_palette = [
    "#7BB44F",
    "#15A6EC",
    "#F5C02C",
    "#EE2926",
    "#961192",
    "#16325C",
]

# 2000s palette from GraphPad Prism
prism_2000s_palette = [
    "#155BE4",
    "#28CE53",
    "#FF0000",
    "#BE1572",
    "#1D8DEE",
    "#6AA823",
    "#FC4B08",
    "#000000",
]

accent_palette = [
    '#7fc97f',
    '#beaed4',
    '#fdc086',
    '#ffff99',
    '#386cb0',
    '#f0027f',
    '#bf5b17',
    '#666666'
]