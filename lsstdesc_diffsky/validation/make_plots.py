# plot catalog data
import os
import numpy as np
import re
from itertools import zip_longest
import matplotlib.pyplot as plt


def get_subsample(dlen, N=10):

    sample = np.random.sample(dlen)
    fraction = min(float(N) / float(dlen), 1.0)
    index = (sample < fraction)
    Nobj = np.count_nonzero(index)
    print('Selected {} objects'.format(Nobj))

    return Nobj, index


def plot_mag_color(ax_all, dat, label, magbins=50, colbins=30, data_mask=None,
                   qnames=None,
                   frame='rest', fsize=16, axis_label=False, aux_color=None,
                   Nmin=0, lgnd_title='', nlgnd=1):

    mlabel = '$M_{{{}}}$' if frame == 'rest' else '$m_{{{}}}$'

    for ax, q in zip_longest(ax_all.flat, qnames):

        if q is None:
            ax.set_visible(False)
            continue

        if q in dat.colnames:
            mask = np.isfinite(dat[q])
            mask = mask if data_mask is None else mask & data_mask
            qbins = colbins if '-' in q else magbins
            if len(dat[q][mask]) > Nmin:
                if aux_color is None:
                    ax.hist(
                        dat[q][mask],
                        qbins,
                        label=label,
                        histtype='step',
                        density=True)

                else:
                    ax.hist(
                        dat[q][mask],
                        qbins,
                        label=label,
                        histtype='step',
                        density=True,
                        color=aux_color)
            else:
                print('Skipping {} plot with < {} entries'.format(label, Nmin))

        if axis_label:
            b = q.split('_')[-1]
            qlabel = '{}-frame ${}$'.format(frame, b) if '-' in q else mlabel.format(b)
            ax.set_xlabel(qlabel, fontsize=fsize)
            ylabel = '${}$'.format(b) if '-' in b else mlabel.format(b)
            ax.set_ylabel('P({})'.format(ylabel), fontsize=fsize)
            ax.legend(loc='best', ncol=nlgnd, title=lgnd_title)

    return


def plot_y_vs_x(ax, x, y, x_bins, y_bins, xlabel=None, ylabel=None,
                cmap='BuPu', contour=False, alphactr=0.8,
                levels=[.05, .2, .5, .8], cntr_label='', cntr_color='black'):

    qd, xedges, yedges = np.histogram2d(x, y, bins=(x_bins, y_bins), density=True)
    qdmasked = np.ma.masked_where(qd.T == 0.0, qd.T)
    if not contour:
        hs = ax.pcolormesh(xedges, yedges, qdmasked, cmap=cmap)
        cbs = plt.colorbar(hs, ax=ax)
        rlabelcs = '$\\rm{{density}$'
        cbs.set_label(rlabelcs)
    else:
        x_cen, y_cen = np.meshgrid(xedges[:-1], yedges[:-1])
        x_cen += (0.5 * (xedges[1] - xedges[0]))  # find bin centers
        y_cen += (0.5 * (yedges[1] - yedges[0]))
        cnt = ax.contour(
            x_cen,
            y_cen,
            qd.T,
            colors=cntr_color,
            levels=levels,
            alpha=alphactr)
        ax.clabel(cnt, inline=1, fontsize=8)
        cnt.collections[0].set_label(cntr_label)
        # ax.legend(loc='best')

    # ax.set_xscale(xscale)
    # ax.set_yscale(yscale)
    if xlabel is not None:
        ax.set_xlabel('${}$'.format(xlabel))
    if ylabel is not None:
        ax.set_ylabel('${}$'.format(ylabel))
    # ax.set_title('t = {:.2f} Gyr, z={:.2f}'.format(t, z))

    return


def plot_color_color(ax_all, dat, label, magbins=30, colbins=20, data_mask=None,
                     qnames=None,
                     frame='rest', fsize=16, Nmin=0, cmap='viridis',
                     scatterplot=False, aux_color=None,
                     levels=[.05, .2, .5, .8], contour=False, lgnd_title='', nlgnd=1):

    mlabel = 'M_{{{}}}' if frame == 'rest' else 'm_{{{}}}'

    for ax, qy, qx in zip_longest(ax_all.flat, qnames[1:], qnames[:-1]):

        if qx is None or qy is None:
            ax.set_visible(False)
            continue
        if qx not in dat.colnames or qy not in dat.colnames:
            continue

        mask = np.isfinite(dat[qx]) & np.isfinite(dat[qy])
        mask = mask if data_mask is None else mask & data_mask
        y = dat[qy][mask]
        x = dat[qx][mask]
        x_bins = colbins if '-' in qx else magbins
        y_bins = colbins if '-' in qy else magbins

        xlabel = qx.split('_')[-1] if '-' in qx else mlabel.format(qx)
        ylabel = qy.split('_')[-1]
        # print(qx,qy, len(x), len(y))
        if len(y) > Nmin and len(x) > Nmin:
            if not scatterplot:
                plot_y_vs_x(ax, x, y, x_bins, y_bins, xlabel=xlabel, ylabel=ylabel,
                            cmap=cmap,
                            levels=levels, contour=contour, cntr_color=aux_color)
            else:
                ax.scatter(x, y, s=10, c=aux_color, label=label)
                ax.legend(loc='best', ncol=nlgnd, title=lgnd_title)
        else:
            print('Skipping {}, #x/#y={}/{} < {}'.format(label, len(x), len(y), Nmin))

    return


def get_nrow_ncol(nq, nrow=2):

    nrow = nrow if nq <= 6 else 3
    nrow = 1 if nq < 4 else nrow
    nrow = 4 if nq > 12 else nrow
    ncol = int(np.ceil(float(nq) / float(nrow)))

    return nrow, ncol


def save_fig(fig, plotdir, pltname):
    figname = os.path.join(plotdir, pltname)
    fig.savefig(figname)  # , bbox_inches='tight')
    print('Saving {}'.format(figname))


def fix_plotid(plotid):
    plotid = re.sub(r"_\*", '', plotid)
    plotid = re.sub(r"\*", '', plotid)
    plotid = '_' + plotid if len(plotid) > 0 and plotid[0] != '_' else plotid
    return plotid


zbins = np.linspace(0, 1.5, 150)
colorbins = 200


def plot_color_redshift(t, zbins, colorbins, filters, frames=['rest', 'obs'],
                        plotid='', plotdir='plots/colors', fsize=16, z='redshift',
                        scatter=False, zsnaps=None,
                        pltname='color_z_{}_{}{}.png', cmap='viridis', Nsub=10000):

    plotid = plotid + '_scatt' if scatter else plotid
    for f in filters:
        for fr in frames:
            colorlist = [c for c in t.colnames if f in c and '-' in c and fr in c]
            nrow, ncol = get_nrow_ncol(len(colorlist))

            fig, ax_all = plt.subplots(nrow, ncol, figsize=(ncol * 7, nrow * 5))
            for ax, col in zip_longest(ax_all.flat, colorlist):
                if col is not None:
                    maskz = (t[z] <= zbins[-1])
                    col_bins = np.linspace(
                        np.min(
                            t[col][maskz]), np.max(
                            t[col][maskz]), colorbins)
                    if zsnaps is not None:
                        for zs in zsnaps[(zsnaps <= zbins[-1])]:
                            ax.axvline(zs, color='r', lw=0.7)
                    if scatter:
                        Ngals, index = get_subsample(np.count_nonzero(maskz), N=Nsub)
                        ax.scatter(t[z][maskz][index], t[col][maskz][index], s=1)
                    else:
                        plot_y_vs_x(
                            ax,
                            t[z][maskz],
                            t[col][maskz],
                            zbins,
                            col_bins,
                            cmap=cmap)

                    ax.set_xlabel('$z$', size=fsize)
                    ax.set_ylabel(col.split('_')[-1], size=fsize)
                else:
                    ax.set_visible(False)

        fig.suptitle('{} {}-frame'.format(f, fr), fontsize=fsize + 3)
        plotid = fix_plotid(plotid)
        save_fig(fig, plotdir, pltname.format(f, fr, plotid))

    return


def plot_q_redshift(t, qlist, zbins, Nqbins, title='',
                    plotid='', plotdir='plots/colors', fsize=16, z='redshift',
                    scatter=False,
                    pltname='{}_z_{}.png', cmap='viridis', Nsub=10000):

    plotid = plotid + '_scatt' if scatter else plotid
    nrow, ncol = get_nrow_ncol(len(qlist))
    fig, ax_all = plt.subplots(nrow, ncol, figsize=(ncol * 7, nrow * 5))
    for ax, q in zip_longest(ax_all.flat, qlist):
        if q is not None:
            maskz = (t[z] <= zbins[-1])
            q_bins = np.linspace(np.min(t[q][maskz]), np.max(t[q][maskz]),
                                 Nqbins)
            if scatter:
                Ngals, index = get_subsample(np.count_nonzero(maskz), N=Nsub)
                ax.scatter(t[z][maskz][index], t[q][maskz][index], s=1)
            else:
                plot_y_vs_x(ax, t[z][maskz], t[q][maskz], zbins, q_bins,
                            cmap=cmap)

                ax.set_xlabel('$z$', size=fsize)
                ax.set_ylabel(q.split('_')[-1], size=fsize)
        else:
            ax.set_visible(False)

    fig.suptitle('{} {}'.format(q, title), fontsize=fsize + 3)
    plotid = fix_plotid(plotid)
    save_fig(fig, plotdir, pltname.format(q, plotid))

    return
