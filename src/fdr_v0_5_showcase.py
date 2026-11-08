# %% [markdown]
# FDR Conceptual Simulation v0.5 — Showcase Edition
#
# W-type 3-basin landscapes with different barrier heights × Cue OFF/ON (6 panels),
# plus an optional explicit diffuse / unbound stress condition.
#
# Scope: a transparent toy model illustrating possible state-space behaviors.
# NOT a biological memory model, an empirical validation, or evidence that a
# particular neural system has this potential landscape.
#
# Outputs:
# - FDR_v0_5_terrain_3d.png
# - FDR_v0_5_standard_6panel.mp4, FDR_v0_5_standard_final.png
# - FDR_v0_5_diffuse_stress_6panel.mp4, FDR_v0_5_diffuse_stress_final.png
# - FDR_v0_5_timecourses.png, metrics.csv, summary.csv, parameters.json
#
# How to tune:
# - edit BARRIER_LEVELS for the three showcased landscapes,
# - edit CUE_STRENGTH, RECOVERY_DIFFUSION_STRESS, SEED etc.,
# - or launch the optional ipywidgets barrier explorer in the Notebook.

# %% [markdown]
# 1. Parameters and experimental design
#
# The 3D terrain uses a shared vertical scale and display clipping at
# SURFACE_DISPLAY_MAX for clarity; all particle dynamics use the *unclipped* U.
# Distinct low/medium/high barriers share the same basin minima and trench shape.

# %%
from __future__ import annotations
from pathlib import Path
import csv
import json
import shutil
import warnings
import subprocess
from PIL import Image, ImageDraw, ImageFont

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

SEED = 20261107
N_PARTICLES = 180
DT = 0.035
T_END = 10.5
T_FLOW_END = 2.0
T_DISSIPATION_END = 6.0

L = 2.20
H = 0.75
BETA_TRENCH = 1.10
GAMMA_TRENCH = 0.050
BARRIER_LEVELS = {"LOW": 0.012, "MEDIUM": 0.030, "HIGH": 0.072}
CUE_STRENGTH = 0.46
CUE_RAMP = 0.60
D_BASE = 0.014
D_PEAK = 0.62
D_RAMP_UP = 0.85
D_RAMP_DOWN = 0.90

# Optional v0.4-inspired stress protocol: continuing diffusion and weakened
# basin restoring force, with a small escape drift away from the trench.
RECOVERY_DIFFUSION_STRESS = 0.34
STRESS_RESTORING_FRACTION = 0.22
STRESS_ESC_DRIFT_STRENGTH = 1.0

BASIN_RADIUS = 0.52  # operational metric only, not a true stochastic basin boundary
FPS = 12
FRAME_STRIDE = 5
SAVE_MP4 = True
SAVE_STRESS_MP4 = True
SURFACE_DISPLAY_MAX = 2.40

OUT_DIR = Path("outputs") / "showcase_v0.5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Clear phase naming; time is dimensionless and not physical seconds.
SCENARIOS = ("standard", "diffuse_stress")
CUE_STATES = (False, True)
COLORS = {False: "#ffb57d", True: "#5ee4cf"}
BG = "#111923"
TEXT = "#eef4f6"
ACCENT = "#ffde8a"

# %% [markdown]
# 2. The W-type 3-basin potential and stochastic dynamics
#
# U(x,y;alpha) = alpha*x²*(x²-L²)² + beta*(y-y_c(x))² + gamma*(y-y_c(x))⁴
# y_c(x) = H*cos(pi*x/L)
#
# Along this specified W-shaped trench centerline, the barrier between the
# minima is Delta_U_trench = 4*alpha*L⁶/27. This is not claimed to be the
# exact minimum-energy saddle barrier across all paths in the full 2D surface.
#
# dX_t = [ -kappa(t)*grad U(X_t) + cue(t,X_t) + escape(t,X_t) ] dt
#      + sqrt(2*D(t)*g(U(X_t))) dW_t
#
# This is an Itô Euler–Maruyama discretization. The same standard normal
# variates and initial coordinates are reused across every comparison panel.
# Multiplicative diffusion can produce different realized increments despite
# common random numbers; this is intentional and made explicit.

# %%
def y_center(x):
    return H * np.cos(np.pi * x / L)


def y_center_prime(x):
    return -(np.pi * H / L) * np.sin(np.pi * x / L)


BASIN_CENTERS = np.array([[-L, -H], [0.0, H], [L, -H]], dtype=float)
TARGET = BASIN_CENTERS[0].copy()


def potential(xy: np.ndarray, alpha: float) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    x, y = xy[..., 0], xy[..., 1]
    s = y - y_center(x)
    return alpha * x*x*(x*x-L*L)**2 + BETA_TRENCH*s*s + GAMMA_TRENCH*s**4


def potential_gradient(xy: np.ndarray, alpha: float) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    x, y = xy[..., 0], xy[..., 1]
    s, yc_prime = y-y_center(x), y_center_prime(x)
    gx = alpha*(2*x*(x*x-L*L)*(3*x*x-L*L)) - (2*BETA_TRENCH*s + 4*GAMMA_TRENCH*s**3)*yc_prime
    gy = 2*BETA_TRENCH*s + 4*GAMMA_TRENCH*s**3
    return np.stack((gx, gy), axis=-1)


def trench_barrier(alpha: float) -> float:
    """Barrier height measured along y=y_center(x), not a full 2D saddle proof."""
    return 4 * alpha * L**6 / 27


def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u*u*(3 - 2*u)


def cue_activation(t):
    return smoothstep((t-T_DISSIPATION_END)/CUE_RAMP)


def diffusion_standard(t):
    rise = smoothstep((t-T_FLOW_END)/D_RAMP_UP)
    fall = 1 - smoothstep((t-(T_DISSIPATION_END-D_RAMP_DOWN))/D_RAMP_DOWN)
    return D_BASE + (D_PEAK-D_BASE)*rise*fall


def diffusion_factor(xy, alpha):
    """Same U-dependent diffusion law in all 6 panels; positive scalar."""
    u = potential(xy, alpha)
    return 0.80 + 0.70/(1 + np.exp(-np.clip((u-0.10)/0.45, -60, 60)))


def diffuse_escape(xy):
    """Explicit *stress* intervention, not a spontaneous emergent outcome."""
    x, y = xy[:,0], xy[:,1]
    outside = (np.abs(y-y_center(x)) > 0.55)[:,None]
    displacement = xy[:,None,:]-BASIN_CENTERS[None,:,:]
    radius = np.sqrt(np.sum(displacement**2, axis=-1)) + 0.08
    repulsion = np.sum(displacement/(radius[...,None]**2), axis=1)
    sea_push = np.column_stack((0.05*np.sign(x), 0.12+0.05*np.tanh(y)))
    return outside*(0.18*repulsion + sea_push)


def nearest_and_bound(xy):
    d = np.linalg.norm(xy[:,None,:]-BASIN_CENTERS[None,:,:],axis=-1)
    labels = np.argmin(d,axis=1)
    bound = np.min(d,axis=1) <= BASIN_RADIUS
    return labels, bound


def fractions(xy):
    labels,bound = nearest_and_bound(xy)
    x = [np.mean(bound & (labels==i)) for i in range(3)]
    return np.array([*x, np.mean(~bound)], dtype=float)  # A/B/C/unbound, sum=1


def dispersion(xy):
    """Trace of population covariance of state-coordinate samples."""
    return float(np.sum(np.var(xy, axis=0)))


def simulate_pair(alpha, scenario, initial, shared_gaussian, times):
    """
    Returns trajectories [cue=off/on, step, particle, xy].
    Every pair shares the same initial states and standard normal innovations.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario}")
    steps = len(times)-1
    paths = np.empty((2, len(times), len(initial), 2), dtype=float)
    paths[:,0,:,:] = initial
    for k in range(steps):
        t = times[k]
        activated = cue_activation(t)
        diff = diffusion_standard(t)
        if scenario == "diffuse_stress":
            diff = (1-activated)*diff + activated*RECOVERY_DIFFUSION_STRESS
            restoring = (1-activated) + activated*STRESS_RESTORING_FRACTION
        else:
            restoring = 1.0
        for cue_idx, cue_on in enumerate(CUE_STATES):
            current = paths[cue_idx,k,:,:]
            drift = -restoring*potential_gradient(current,alpha)
            if scenario == "diffuse_stress":
                drift += activated*STRESS_ESC_DRIFT_STRENGTH*diffuse_escape(current)
            if cue_on:
                drift += activated*CUE_STRENGTH*(TARGET-current)
            factor = diffusion_factor(current,alpha)
            sigma = np.sqrt(2*diff*factor*DT)[:,None]
            paths[cue_idx,k+1,:,:] = current + DT*drift + sigma*shared_gaussian[k]
    if not np.isfinite(paths).all():
        raise FloatingPointError("Numerical instability: adjust DT/terrain parameters")
    return paths


def simulate_suite():
    rng = np.random.default_rng(SEED)
    n_steps = int(round(T_END/DT))
    times = np.arange(n_steps+1)*DT
    initial = TARGET + rng.normal(0,0.12,size=(N_PARTICLES,2))
    shared_gaussian = rng.standard_normal((n_steps,N_PARTICLES,2))
    trajectories = {
        scenario: {
            name: simulate_pair(alpha,scenario,initial,shared_gaussian,times)
            for name,alpha in BARRIER_LEVELS.items()
        }
        for scenario in SCENARIOS
    }
    return times,trajectories,initial,shared_gaussian

# %% [markdown]
# 3. Data export, numerical checks and diagnostics
#
# Readouts: operational A/B/C (within radius) and "unbound" (outside all
# disks). Unbound is finite-time, not permanent erasure or loss of memory.

# %%
def save_diagnostics(times,trajectories):
    summary_file = OUT_DIR/"FDR_v0_5_summary.csv"
    time_file = OUT_DIR/"FDR_v0_5_timecourse.csv"
    with summary_file.open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f)
        writer.writerow(["scenario","barrier","alpha","barrier_along_trench","cue","A","B","C","unbound","dispersion","seed","dt","particles"])
        for scenario in SCENARIOS:
            for name,alpha in BARRIER_LEVELS.items():
                for ci,cue in enumerate(CUE_STATES):
                    xy = trajectories[scenario][name][ci,-1]
                    a,b,c,u = fractions(xy)
                    writer.writerow([scenario,name,alpha,trench_barrier(alpha),"ON" if cue else "OFF",a,b,c,u,dispersion(xy),SEED,DT,N_PARTICLES])
    with time_file.open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f)
        writer.writerow(["scenario","barrier","alpha","cue","t","A","B","C","unbound","dispersion"])
        for scenario in SCENARIOS:
            for name,alpha in BARRIER_LEVELS.items():
                for ci,cue in enumerate(CUE_STATES):
                    for k in range(0,len(times),FRAME_STRIDE):
                        xy=trajectories[scenario][name][ci,k]
                        writer.writerow([scenario,name,alpha,"ON" if cue else "OFF",times[k],*fractions(xy),dispersion(xy)])
    params={
        "title":"FDR Conceptual Simulation v0.5",
        "status":"toy conceptual demonstration; no neuroscience validation",
        "seed":SEED,"particles":N_PARTICLES,"dt":DT,"t_end":T_END,
        "flow_end":T_FLOW_END,"dissipation_end":T_DISSIPATION_END,
        "L":L,"H":H,"BETA_TRENCH":BETA_TRENCH,"GAMMA_TRENCH":GAMMA_TRENCH,
        "barrier_levels_alpha":BARRIER_LEVELS,
        "barrier_along_W_trench":{n:trench_barrier(a) for n,a in BARRIER_LEVELS.items()},
        "cue_strength":CUE_STRENGTH,"D_base":D_BASE,"D_peak":D_PEAK,
        "D_stress":RECOVERY_DIFFUSION_STRESS,"stress_restoring_fraction":STRESS_RESTORING_FRACTION,
        "stress_escape_drift_strength":STRESS_ESC_DRIFT_STRENGTH,
        "basin_radius":BASIN_RADIUS,"animation_stride":FRAME_STRIDE,"fps":FPS,
        "scenarios":SCENARIOS,"common_random_numbers":True,
        "metric_definition":"A/B/C=within radius of nearest of three minima; unbound=outside all three circles",
    }
    (OUT_DIR/"FDR_v0_5_parameters.json").write_text(json.dumps(params,ensure_ascii=False,indent=2),encoding="utf-8")
    return summary_file,time_file


def check_model():
    for name,alpha in BARRIER_LEVELS.items():
        err = np.max(np.abs(potential_gradient(BASIN_CENTERS,alpha)))
        if err>1e-8:
            raise AssertionError(f"stationary-point check failed for {name}: {err}")
        # Numerical differentiation of gradient away from minima
        x=np.array([[0.37,-0.23],[1.38,0.45]])
        eps=1e-6
        grad_num=np.column_stack([
            (potential(x+np.eye(2)[i]*eps,alpha)-potential(x-np.eye(2)[i]*eps,alpha))/(2*eps)
            for i in range(2)
        ])
        if not np.allclose(potential_gradient(x,alpha),grad_num,atol=1e-6,rtol=1e-6):
            raise AssertionError("gradient finite-difference check failed")
    return True

# %% [markdown]
# 4. Visual assets: shared-height 3D terrain thumbnails + six 2D particle panels
#
# The 3D terrain row deliberately uses the *same scale* to make LOW vs HIGH
# visibly different. Surface clipping is for display only, never for dynamics.

# %%
def make_terrain_thumbnail(alpha):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    xs=np.linspace(-2.65,2.65,85)
    ys=np.linspace(-1.55,1.55,62)
    XX,YY=np.meshgrid(xs,ys)
    XY=np.stack((XX,YY),axis=-1)
    Z=np.minimum(potential(XY,alpha),SURFACE_DISPLAY_MAX)
    fig=plt.figure(figsize=(5.0,3.0),dpi=110,facecolor=BG)
    ax=fig.add_subplot(111,projection="3d")
    ax.set_facecolor(BG)
    ax.plot_surface(XX,YY,Z,cmap="cividis",norm=Normalize(0,SURFACE_DISPLAY_MAX),
                    alpha=.92,rstride=2,cstride=2,linewidth=0,antialiased=True)
    xp=np.linspace(-L,L,140)
    yp=y_center(xp)
    zp=potential(np.stack((xp,yp),axis=-1),alpha)+.06
    ax.plot(xp,yp,zp,color="#ffe6a3",lw=2.5,alpha=1)
    for i,letter in enumerate("ABC"):
        p=BASIN_CENTERS[i]
        ax.text(p[0],p[1],.12,letter,color="white",fontsize=10,weight="bold")
    ax.set_xlim(-2.7,2.7)
    ax.set_ylim(-1.6,1.6)
    ax.set_zlim(0,SURFACE_DISPLAY_MAX)
    ax.view_init(elev=25,azim=-70)
    ax.set_box_aspect((1.7,1.0,.7),zoom=.98)
    ax.set_axis_off()
    fig.subplots_adjust(0,0,1,1)
    fig.canvas.draw()
    pixels=np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return pixels


def make_terrain_thumbnails():
    return {name:make_terrain_thumbnail(alpha) for name,alpha in BARRIER_LEVELS.items()}


def plot_terrain_triptych(thumbnails):
    fig,axes=plt.subplots(1,3,figsize=(15.5,4.2),dpi=110,facecolor=BG)
    for ax,(name,alpha) in zip(axes,BARRIER_LEVELS.items()):
        ax.imshow(thumbnails[name])
        ax.set_axis_off()
        ax.set_title(f"{name}   barrier along trench = {trench_barrier(alpha):.2f}",
                     color=TEXT,pad=3,fontsize=12,weight="bold")
    fig.suptitle("FDR v0.5 | W-TYPE 3-BASIN LANDSCAPES | SHARED HEIGHT SCALE",color=TEXT,y=.99,fontsize=15,weight="bold")
    fig.text(.5,.02,"Displayed heights are clipped uniformly at U = 2.4; simulated forces use the unclipped potential.",ha="center",color="#afbfcd",fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.95),w_pad=.1)
    p=OUT_DIR/"FDR_v0_5_terrain_3d.png"
    fig.savefig(p,facecolor=BG,bbox_inches="tight")
    plt.close(fig)
    return p


def six_panel_figure(times,scenario_paths,scenario,thumbnails,frame_index=-1,animation=False):
    """Make a 3D-terrain header and six overview panels with common map scale."""
    xs=np.linspace(-2.85,2.85,120)
    ys=np.linspace(-1.75,1.75,84)
    XX,YY=np.meshgrid(xs,ys)
    fig=plt.figure(figsize=(15.6,11.2),dpi=100,facecolor=BG)
    grid=GridSpec(3,3,figure=fig,height_ratios=[.93,1.08,1.08],
                  hspace=.34,wspace=.12,left=.054,right=.988,bottom=.055,top=.905)
    axes_top=[fig.add_subplot(grid[0,col]) for col in range(3)]
    for i,(name,alpha) in enumerate(BARRIER_LEVELS.items()):
        ax=axes_top[i]
        ax.imshow(thumbnails[name],interpolation="bilinear")
        ax.set_axis_off()
        ax.set_title(f"{name}   |  barrier height ≈ {trench_barrier(alpha):.2f}",
                     color=ACCENT,fontsize=11,weight="bold",pad=2)
    axes={}
    scatters={}
    footer_texts={}
    for row,cue_on in enumerate(CUE_STATES):
        for col,(name,alpha) in enumerate(BARRIER_LEVELS.items()):
            ax=fig.add_subplot(grid[row+1,col])
            U=potential(np.stack((XX,YY),axis=-1),alpha)
            lev=np.linspace(0,3,22)
            ax.contourf(XX,YY,U,levels=lev,cmap="cividis",extend="max",vmin=0,vmax=3)
            ax.contour(XX,YY,U,levels=np.linspace(.10,2.8,11),colors="white",alpha=.16,linewidths=.5)
            xp=np.linspace(-L,L,160)
            ax.plot(xp,y_center(xp),color="#f6eab9",lw=1.3,alpha=.85)
            ax.scatter(BASIN_CENTERS[:,0],BASIN_CENTERS[:,1],s=26,c="white",marker="x",lw=.9)
            for bi,label in enumerate("ABC"):
                p=BASIN_CENTERS[bi]
                ax.text(p[0]+.10,p[1]+.10,label,color="white",fontsize=9,weight="bold")
            ax.set_xlim(-2.85,2.85)
            ax.set_ylim(-1.75,1.75)
            ax.set_aspect('equal', adjustable='box')
            ax.set_facecolor(BG)
            ax.tick_params(colors="#b5c4cf",labelsize=7)
            ax.set_xlabel("state x",color="#aebac2",fontsize=8)
            if col==0:
                ax.set_ylabel("state y",color="#aebac2",fontsize=8)
            ax.set_title("CUE OFF" if not cue_on else "CUE ON",color=COLORS[cue_on],fontsize=10,weight="bold",pad=4)
            dots=ax.scatter([],[],s=12,c=COLORS[cue_on],alpha=.96,edgecolors="none")
            txt=ax.text(.5,.025,"",transform=ax.transAxes,ha="center",va="bottom",
                        color=COLORS[cue_on],fontsize=8.4,
                        bbox=dict(facecolor=BG, alpha=.8, edgecolor="none", pad=2))
            axes[(name,cue_on)]=ax
            scatters[(name,cue_on)]=dots
            footer_texts[(name,cue_on)]=txt
    heading=fig.suptitle("",color=TEXT,weight="bold",fontsize=16,y=.973)
    fig.text(.5,.026,
             "Same initial states + Gaussian noise | same cue field within each scenario | barriers vary only through alpha",
             color="#bdc9d3",fontsize=8.3,ha="center")
    # Figure should have enough vertical margin for captions and be clear on phones in landscape.
    def update(k):
        t=times[k]
        stage="FLOW" if t<T_FLOW_END else "DISSIPATION" if t<T_DISSIPATION_END else "OUTCOME"
        scenario_label="BASELINE" if scenario=="standard" else "DIFFUSE STRESS (EXPLICIT INTERVENTION)"
        heading.set_text(f"FDR v0.5  |  {scenario_label}  |  {stage}  |  t={t:.1f}")
        for name in BARRIER_LEVELS:
            pair=scenario_paths[name]
            for ci,cue_on in enumerate(CUE_STATES):
                xy=pair[ci,k]
                scatters[(name,cue_on)].set_offsets(xy)
                a,b,c,u=fractions(xy)
                footer_texts[(name,cue_on)].set_text(f"A {a:.0%}   B {b:.0%}   C {c:.0%}   unbound {u:.0%}")
        return [heading,*scatters.values(),*footer_texts.values()]
    update(frame_index)
    return fig,update,axes,scatters,footer_texts,heading


def save_six_panel_snapshot(times,scenario_paths,scenario,thumbnails):
    fig,*_=six_panel_figure(times,scenario_paths,scenario,thumbnails,frame_index=-1)
    p=OUT_DIR/f"FDR_v0_5_{scenario}_final.png"
    fig.savefig(p,facecolor=BG,bbox_inches="tight",dpi=110)
    plt.close(fig)
    return p


def _load_demo_font(size, bold=False):
    # Font is used in the render environment but is never embedded or distributed.
    path=("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
          else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def save_six_panel_movie(times,scenario_paths,scenario,thumbnails):
    """
    Render a static six-panel terrain once, then composite just the moving
    particles / readouts with Pillow. Avoids repeatedly drawing 3D/contours.
    ffmpeg receives RGB frames over stdin; no temporary frame images required.
    """
    if not shutil.which("ffmpeg"):
        warnings.warn("ffmpeg not found: use Colab or install ffmpeg for MP4 export")
        return None
    fig,update,axes,scatters,footers,heading=six_panel_figure(
        times,scenario_paths,scenario,thumbnails,frame_index=0,animation=True
    )
    # Freeze all static background features and remove the dynamic artists.
    heading.set_text("")
    for dot in scatters.values():
        dot.set_offsets(np.empty((0,2)))
    for footer in footers.values():
        footer.set_visible(False)
    fig.canvas.draw()
    canvas=np.asarray(fig.canvas.buffer_rgba())
    background=Image.fromarray(canvas).convert("RGB")
    W,Hpx=background.size
    pixinfo={}
    for (name,cue_on),ax in axes.items():
        bbox=ax.bbox
        pixinfo[(name,cue_on)]={
            "transform":ax.transData,
            "clip":(bbox.x0,Hpx-bbox.y1,bbox.x1,Hpx-bbox.y0),
            "label_pos":(0.5*(bbox.x0+bbox.x1),Hpx-bbox.y0-9),
        }
    plt.close(fig)

    font_header=_load_demo_font(18,bold=True)
    font_label=_load_demo_font(12,bold=True)
    indices=np.unique(np.r_[np.arange(0,len(times),FRAME_STRIDE),len(times)-1])
    p=OUT_DIR/f"FDR_v0_5_{scenario}_6panel.mp4"
    cmd=["ffmpeg","-y","-loglevel","error","-f","rawvideo","-vcodec","rawvideo",
         "-pixel_format","rgb24","-video_size",f"{W}x{Hpx}","-framerate",str(FPS),
         "-i","-","-an","-c:v","libx264","-preset","veryfast","-crf","22",
         "-pix_fmt","yuv420p","-movflags","+faststart",str(p)]
    process=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    try:
        for k in indices:
            frame=background.copy()
            draw=ImageDraw.Draw(frame)
            t=times[int(k)]
            stage="FLOW" if t<T_FLOW_END else "DISSIPATION" if t<T_DISSIPATION_END else "OUTCOME"
            scenario_name="BASELINE" if scenario=="standard" else "DIFFUSE STRESS"
            title=f"FDR v0.5  |  {scenario_name}  |  {stage}  |  t={t:.1f}"
            draw.text((W//2,24),title,font=font_header,fill="white",anchor="mm")
            for name in BARRIER_LEVELS:
                for ci,cue_on in enumerate(CUE_STATES):
                    xy=scenario_paths[name][ci,int(k)]
                    spec=pixinfo[(name,cue_on)]
                    xy_pix=spec["transform"].transform(xy)
                    r=2.1
                    left,top,right,bottom=spec["clip"]
                    point_color=(255,181,125) if not cue_on else (94,228,207)
                    for x,y in xy_pix:
                        py=Hpx-y
                        if left+2 < x < right-2 and top+2 < py < bottom-2:
                            draw.ellipse((x-r,py-r,x+r,py+r),fill=point_color)
                    vals=fractions(xy)
                    msg=f"A {vals[0]:.0%}    B {vals[1]:.0%}    C {vals[2]:.0%}    unbound {vals[3]:.0%}"
                    cx,cy=spec["label_pos"]
                    # Dark rounded label inside the panel, clear even over yellow regions.
                    bb=draw.textbbox((cx,cy),msg,font=font_label,anchor="mm")
                    pad=4
                    draw.rounded_rectangle((bb[0]-pad,bb[1]-pad,bb[2]+pad,bb[3]+pad),
                                           radius=4,fill=(17,25,35))
                    draw.text((cx,cy),msg,font=font_label,
                              fill=point_color,anchor="mm")
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        stderr=process.stderr.read()
        rc=process.wait()
        if rc!=0:
            raise RuntimeError(f"ffmpeg exited with status {rc}: {stderr.decode('utf-8','replace')[:800]}")
    except Exception:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        process.kill()
        process.wait()
        raise
    finally:
        if process.stderr:
            process.stderr.close()
    return p


def save_timecourse(times,trajectories):
    fig,axes=plt.subplots(2,2,figsize=(13.6,7.8),dpi=120,sharex=True,facecolor="white")
    cmap={"LOW":"#2f9f9f","MEDIUM":"#c2912b","HIGH":"#ac5e8c"}
    for row,scenario in enumerate(SCENARIOS):
        for name in BARRIER_LEVELS:
            for ci,cue_on in enumerate(CUE_STATES):
                arr=trajectories[scenario][name][ci]
                metrics=np.array([fractions(xy) for xy in arr])
                label=f"{name} {'Cue ON' if cue_on else 'Cue OFF'}"
                ls="-" if cue_on else "--"
                axes[row,0].plot(times,metrics[:,0],label=label,color=cmap[name],lw=1.5,ls=ls)
                axes[row,1].plot(times,metrics[:,3],label=label,color=cmap[name],lw=1.5,ls=ls)
        for col in range(2):
            ax=axes[row,col]
            ax.axvline(T_FLOW_END,c="#777",ls=":",lw=1)
            ax.axvline(T_DISSIPATION_END,c="#777",ls=":",lw=1)
            ax.set_ylim(0,1.02)
            ax.grid(alpha=.2)
            ax.set_ylabel("fraction")
            if row==1:
                ax.set_xlabel("model time (arbitrary units)")
        axes[row,0].set_title(f"{scenario}: within Basin A",loc="left",fontsize=11)
        axes[row,1].set_title(f"{scenario}: unbound",loc="left",fontsize=11)
    axes[0,1].legend(loc="center left",bbox_to_anchor=(1.01,.5),fontsize=8,frameon=False)
    fig.suptitle("FDR v0.5 | A-region and unbound timecourses | solid = Cue ON",fontsize=14,weight="bold")
    fig.tight_layout(rect=(0,0,.84,.955))
    p=OUT_DIR/"FDR_v0_5_timecourses.png"
    fig.savefig(p,dpi=120,bbox_inches="tight")
    plt.close(fig)
    return p

# %% [markdown]
# 5. Interactive barrier slider (Colab / Jupyter, optional)
#
# This explorer reuses a fixed initial state/noise sequence and recomputes a
# *single* alpha and Cue combination; it does not change the predefined six
# baseline panels. To change the six showcase heights, edit BARRIER_LEVELS.

# %%
def launch_interactive_explorer():
    try:
        import ipywidgets as widgets
        from IPython.display import display, clear_output
    except ImportError:
        print("ipywidgets is unavailable. Edit BARRIER_LEVELS in the parameter cell instead.")
        return

    try:
        from google.colab import output
        output.enable_custom_widget_manager()
    except Exception:
        pass

    alpha_slider=widgets.FloatSlider(value=BARRIER_LEVELS["MEDIUM"],min=.006,max=.090,step=.003,
                                     description="Barrier α",readout_format=".3f",continuous_update=False)
    cue_toggle=widgets.Checkbox(value=True,description="Cue ON")
    scenario_choice=widgets.Dropdown(options=[("Baseline","standard"),("Diffuse stress","diffuse_stress")],
                                     value="standard",description="Regime")
    run_button=widgets.Button(description="Run this setting",button_style="info",icon="play")
    panel=widgets.Output()

    # Once generated, the fixed Gaussian samples do not depend on alpha.
    rng=np.random.default_rng(SEED)
    n_steps=int(round(T_END/DT))
    tt=np.arange(n_steps+1)*DT
    ini=TARGET+rng.normal(0,.12,(N_PARTICLES,2))
    shared=rng.standard_normal((n_steps,N_PARTICLES,2))

    def on_run(_):
        alpha=float(alpha_slider.value)
        scenario=scenario_choice.value
        ci=1 if cue_toggle.value else 0
        pair=simulate_pair(alpha,scenario,ini,shared,tt)
        final=pair[ci,-1]
        with panel:
            clear_output(wait=True)
            fig,axes=plt.subplots(1,2,figsize=(11,3.8))
            xx=np.linspace(-2.85,2.85,150)
            yy=np.linspace(-1.7,1.7,100)
            XX,YY=np.meshgrid(xx,yy)
            U=potential(np.stack((XX,YY),axis=-1),alpha)
            axes[0].contourf(XX,YY,U,levels=np.linspace(0,3.0,25),cmap="cividis",extend="max")
            axes[0].scatter(final[:,0],final[:,1],s=7,c="#55d7ce")
            axes[0].set(xlim=(-2.85,2.85),ylim=(-1.7,1.7),title="Final sample positions",xlabel="state x",ylabel="state y")
            axes[0].set_aspect('equal')
            ux=np.linspace(-L,L,300)
            u_line=potential(np.stack((ux,y_center(ux)),axis=-1),alpha)
            axes[1].plot(ux,u_line,c="#b66c3d",lw=2.5)
            axes[1].set(xlim=(-L,L),ylim=(0,1.65),title=f"Along-trench barrier = {trench_barrier(alpha):.2f}",xlabel="state x",ylabel="U along W corridor")
            vals=fractions(final)
            fig.suptitle(f"{scenario} | {'Cue ON' if ci else 'Cue OFF'} | A/B/C/unbound = "+" / ".join(f"{v:.0%}" for v in vals))
            fig.tight_layout()
            display(fig)
            plt.close(fig)
    run_button.on_click(on_run)
    display(widgets.VBox([widgets.HBox([alpha_slider,cue_toggle,scenario_choice,run_button]),panel]))
    on_run(None)

# %% [markdown]
# 6. Produce publication-ready conceptual-demo assets
#
# Run all cells to compute both regimes, save 3D/6-panel visuals, numerical
# summaries, two MP4s (when ffmpeg is available), and the parameter manifest.

# %%
def main():
    check_model()
    times,trajectories,initial,shared_gaussian=simulate_suite()
    th=make_terrain_thumbnails()
    files=[plot_terrain_triptych(th)]
    files.append(save_timecourse(times,trajectories))
    files.extend(save_diagnostics(times,trajectories))
    for scenario in SCENARIOS:
        files.append(save_six_panel_snapshot(times,trajectories[scenario],scenario,th))
        if SAVE_MP4 and (scenario!="diffuse_stress" or SAVE_STRESS_MP4):
            files.append(save_six_panel_movie(times,trajectories[scenario],scenario,th))
    print("\nFDR v0.5 complete | files:")
    for p in files:
        if p is not None:
            print(f"  {p}   ({p.stat().st_size / 1024:.1f} KiB)")
    print("\nFinal operational readouts (A/B/C/unbound):")
    for scenario in SCENARIOS:
        for name in BARRIER_LEVELS:
            pair=trajectories[scenario][name]
            for ci,cue_on in enumerate(CUE_STATES):
                fr=fractions(pair[ci,-1])
                print(f"  {scenario:14} {name:6} {'Cue ON ' if cue_on else 'Cue OFF'} = "+"/".join(f"{v:.1%}" for v in fr))
    return files


if __name__ == "__main__":
    main()
