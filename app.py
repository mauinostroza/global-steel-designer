from __future__ import annotations
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from aisc360_master_engine_and_audit_v2 import MasterEngineV2, normative_gap_registry_v2
from aisc360_engine_v8_warping_integrated import AngleMember, AngleSection, BlockShearInput, ChannelMember, ChannelSection, DesignMethod, EffectiveAreaInput, FlexureInput, IShapeMember, ISection, Material, MemberDemand, MemberLengths, ShearInput, Strictness, TeeMember, TeeSection
from section_properties_calculator import IShapeDims, AngleDims, SectionPropertyCalculator

st.set_page_config(page_title="Global Steel Designer", layout="wide", initial_sidebar_state="expanded")

@st.cache_data
def load_profiles():
    return pd.read_csv(Path(__file__).with_name("profiles.csv"))

def n_to_kn(x): return x/1000.0
def nmm_to_knm(x): return x/1e6
def inp(units, mv, iv, lm, li): return st.number_input(lm, value=mv) if units=="kN, mm, MPa" else st.number_input(li, value=iv)
def conv(units, Fy, Fu, Lx, Ly, Lb, Pu, Tu, Mux, Muy, Vux, holes):
    if units=="kip, in, ksi":
        return {"Fy":Fy*6.89476,"Fu":Fu*6.89476,"Lx":Lx*25.4,"Ly":Ly*25.4,"Lb":Lb*25.4,"Pu":Pu*4448.22,"Tu":Tu*4448.22,"Mux":Mux*1.35582e6,"Muy":Muy*1.35582e6,"Vux":Vux*4448.22,"holes_area":holes*(25.4**2)}
    return {"Fy":Fy,"Fu":Fu,"Lx":Lx,"Ly":Ly,"Lb":Lb,"Pu":Pu*1000.0,"Tu":Tu*1000.0,"Mux":Mux*1e6,"Muy":Muy*1e6,"Vux":Vux*1000.0,"holes_area":holes}
def bundle_df(bundle, moment=False):
    if bundle is None: return pd.DataFrame()
    return pd.DataFrame([{"Equation":r.equation,"Description":r.description,"Capacity":nmm_to_knm(r.design_strength) if moment else n_to_kn(r.design_strength),"Demand":nmm_to_knm(r.demand) if moment else n_to_kn(r.demand),"Ratio":r.ratio} for r in bundle.results])
def props_df(sec):
    ks=["area","d","bf","b","tf","t","tw","Ix","Iy","rx","ry","Zx","Zy","Sx","Sy","J","Cw","ro","x_sc","y_sc"]
    return pd.DataFrame([{"Property":k,"Value":getattr(sec,k)} for k in ks if hasattr(sec,k)])

df=load_profiles()
st.markdown("<style>[data-testid='stSidebar'] {background:#0f2d46;} .card {background:#f8fafc; padding:1rem; border:1px solid #d9e2ec; border-radius:12px;}</style>", unsafe_allow_html=True)
st.sidebar.markdown("## 🌍 Global Steel Designer")
st.sidebar.caption("Master Engine v2 + v8")
units=st.sidebar.selectbox("Units",["kN, mm, MPa","kip, in, ksi"])
engine_mode=st.sidebar.radio("Engine mode",["best_effort","strict"])
strictness=Strictness.STRICT if st.sidebar.radio("Normative strictness",["STRICT","PRACTICAL"])=="STRICT" else Strictness.PRACTICAL
family=st.sidebar.selectbox("Family",["I Shape","Channel","Angle","Tee"])
source=st.sidebar.selectbox("Source",["Table","Geometric"])

left,mid,right=st.columns([1.1,1.7,1.1])

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    filters={"I Shape":["W","M","S","HP","IPE","IPN","HEA","HEB","HEM","IN","HN"],"Channel":["C","MC"],"Tee":["WT","MT","ST"],"Angle":[]}
    sec_in=None; row=None
    if source=="Table" and family!="Angle":
        cands=df[df["family"].astype(str).isin(filters[family])]
        name=st.selectbox("Section", cands["name"].tolist())
        row=cands[cands["name"]==name].iloc[0]
    else:
        if family=="I Shape":
            d=inp(units,300.0,11.81,"d (mm)","d (in)")
            bf=inp(units,150.0,5.91,"bf (mm)","bf (in)")
            tf=inp(units,12.0,0.47,"tf (mm)","tf (in)")
            tw=inp(units,8.0,0.31,"tw (mm)","tw (in)")
            if units=="kip, in, ksi": d,bf,tf,tw=d*25.4,bf*25.4,tf*25.4,tw*25.4
            sec_in=SectionPropertyCalculator.i_shape(IShapeDims(d=d,bf=bf,tf=tf,tw=tw,name="Custom I"))
        elif family=="Angle":
            d=inp(units,150.0,5.91,"d (mm)","d (in)")
            b=inp(units,100.0,3.94,"b (mm)","b (in)")
            t=inp(units,10.0,0.39,"t (mm)","t (in)")
            if units=="kip, in, ksi": d,b,t=d*25.4,b*25.4,t*25.4
            sec_in=SectionPropertyCalculator.angle(AngleDims(d=d,b=b,t=t,name="Custom L"))
        else:
            st.warning("Use table for this family.")
            st.stop()
    Fy=inp(units,345.0,50.0,"Fy (MPa)","Fy (ksi)")
    Fu=inp(units,450.0,65.0,"Fu (MPa)","Fu (ksi)")
    Lx=inp(units,6000.0,236.22,"Lx (mm)","Lx (in)")
    Ly=inp(units,6000.0,236.22,"Ly (mm)","Ly (in)")
    Lb=inp(units,3000.0,118.11,"Lb (mm)","Lb (in)")
    Kx=st.number_input("Kx",value=1.0); Ky=st.number_input("Ky",value=1.0); Cb=st.number_input("Cb",value=1.0); U=st.number_input("U",value=1.0,min_value=0.0,max_value=1.0)
    holes=inp(units,0.0,0.0,"Hole deduction area (mm²)","Hole deduction area (in²)")
    Pu=inp(units,150.0,33.72,"Pu (kN)","Pu (kip)")
    Tu=inp(units,0.0,0.0,"Tu (kN)","Tu (kip)")
    Mux=inp(units,250.0,184.39,"Mux (kN·m)","Mux (kip-ft)")
    Muy=inp(units,0.0,0.0,"Muy (kN·m)","Muy (kip-ft)")
    Vux=inp(units,80.0,17.98,"Vux (kN)","Vux (kip)")
    a=inp(units,0.0,0.0,"a stiffeners (mm)","a stiffeners (in)")
    stiff=st.checkbox("Stiffeners",value=False); tfa=st.checkbox("Tension field action",value=False)
    eccx=inp(units,0.0,0.0,"e_x (mm)","e_x (in)")
    eccy=inp(units,0.0,0.0,"e_y (mm)","e_y (in)")
    block_enabled=st.checkbox("Evaluate block shear",value=False)
    if block_enabled:
        Avg=st.number_input("Avg (mm²)",value=1200.0); Avn=st.number_input("Avn (mm²)",value=1000.0); Atg=st.number_input("Atg (mm²)",value=500.0); Atn=st.number_input("Atn (mm²)",value=420.0)
        block_vals={"Avg":Avg,"Avn":Avn,"Atg":Atg,"Atn":Atn,"Ubs":1.0}
    else:
        block_vals={"Avg":0.0,"Avn":0.0,"Atg":0.0,"Atn":0.0,"Ubs":1.0}
    stem_in_tension=st.checkbox("Stem in tension (tees)",value=True)
    st.markdown("</div>", unsafe_allow_html=True)

vals=conv(units,Fy,Fu,Lx,Ly,Lb,Pu,Tu,Mux,Muy,Vux,holes)
a_m=a*25.4 if units=="kip, in, ksi" else a
ex=eccx*25.4 if units=="kip, in, ksi" else eccx
ey=eccy*25.4 if units=="kip, in, ksi" else eccy
engine=MasterEngineV2(mode=engine_mode)
err=None; result=None; section=None

try:
    mat=Material(Fy=vals["Fy"],Fu=vals["Fu"])
    lengths=MemberLengths(Lx=vals["Lx"],Ly=vals["Ly"],Lb=vals["Lb"],Kx=Kx,Ky=Ky)
    demand=MemberDemand(Pu=vals["Pu"],Tu=vals["Tu"],Mux=vals["Mux"],Muy=vals["Muy"],Vux=vals["Vux"])
    flex=FlexureInput(Cb=Cb,stem_in_tension=stem_in_tension,connection_eccentricity_x=ex,connection_eccentricity_y=ey)
    eff=EffectiveAreaInput(U=U,holes_deduction_area=vals["holes_area"])
    shear=ShearInput(a=a_m,stiffeners_present=stiff,tension_field_action=tfa)
    block=BlockShearInput(**block_vals) if block_enabled else None
    if family=="I Shape":
        section = sec_in if sec_in is not None else ISection(name=str(row["name"]),area=float(row["A"]),d=float(row["d"]),bf=float(row.get("bf",0)),tf=float(row.get("tf",0)),tw=float(row.get("tw",0)),Ix=float(row.get("Ix",0)),Iy=float(row.get("Iy",0)),Zx=float(row.get("Zx",0)),Zy=float(row.get("Zy",0)),Sx=float(row.get("Sx",0)),Sy=float(row.get("Sy",0)),rx=float(row.get("rx",0)),ry=float(row.get("ry",0)),J=float(row.get("J",0)),Cw=float(row.get("Cw",0)),ro=float(row.get("ro",0)),rts=float(row.get("rts",0)))
        result = engine.run_i_shape_member(IShapeMember(section=section,material=mat,lengths=lengths,method=DesignMethod.LRFD,strictness=strictness,flexure_input=flex,shear_input=shear,effective_area=eff,block_shear_input=block), demand)
    elif family=="Channel":
        section = ChannelSection(name=str(row["name"]),area=float(row["A"]),d=float(row["d"]),bf=float(row.get("bf",0)),tf=float(row.get("tf",0)),tw=float(row.get("tw",0)),Ix=float(row.get("Ix",0)),Iy=float(row.get("Iy",0)),Zx=float(row.get("Zx",0)),Zy=float(row.get("Zy",0)),Sx=float(row.get("Sx",0)),Sy=float(row.get("Sy",0)),rx=float(row.get("rx",0)),ry=float(row.get("ry",0)),J=float(row.get("J",0)),Cw=float(row.get("Cw",0)),ro=float(row.get("ro",0)))
        result = engine.run_channel_member(ChannelMember(section=section,material=mat,lengths=lengths,method=DesignMethod.LRFD,strictness=strictness,flexure_input=flex,shear_input=shear,effective_area=eff,block_shear_input=block), demand)
    elif family=="Angle":
        section = AngleSection(name=sec_in.name,area=sec_in.area,d=sec_in.d,b=sec_in.bf,t=sec_in.tf,Ix=sec_in.Ix,Iy=sec_in.Iy,rx=sec_in.rx,ry=sec_in.ry,Sx=sec_in.Sx,Sy=sec_in.Sy,Zx=sec_in.Zx,Zy=sec_in.Zy)
        result = engine.run_angle_member(AngleMember(section=section,material=mat,lengths=lengths,method=DesignMethod.LRFD,strictness=strictness,flexure_input=flex,effective_area=eff,block_shear_input=block), demand)
    elif family=="Tee":
        section = TeeSection(name=str(row["name"]),area=float(row["A"]),d=float(row["d"]),bf=float(row.get("bf",0)),tf=float(row.get("tf",0)),tw=float(row.get("tw",0)),Ix=float(row.get("Ix",0)),Iy=float(row.get("Iy",0)),Sx=float(row.get("Sx",0)),Sy=float(row.get("Sy",0)),Zx=float(row.get("Zx",0)),Zy=float(row.get("Zy",0)),rx=float(row.get("rx",0)),ry=float(row.get("ry",0)),J=float(row.get("J",0)),Cw=float(row.get("Cw",0)),ro=float(row.get("ro",0)))
        result = engine.run_tee_member(TeeMember(section=section,material=mat,lengths=lengths,method=DesignMethod.LRFD,strictness=strictness,flexure_input=flex,effective_area=eff,block_shear_input=block), demand)
except Exception as e:
    err=str(e)

with mid:
    st.title("Master Engine Dashboard v2")
    if err:
        st.error(err)
    elif result is not None:
        inter=result.get("interaction_ratio",0.0)
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Section",getattr(section,"name","-")); c2.metric("Family",family); c3.metric("H1 Ratio",f"{inter:.3f}"); c4.metric("Status","✅ Pass" if inter<=1.0 else "❌ Fail")
        tabs=st.tabs(["Results","Audit","Geometry","Gaps Registry"])
        with tabs[0]:
            for key,title,moment in [("tension","Tension",False),("compression","Compression",False),("flexure_major","Flexure Major",True),("flexure_minor","Flexure Minor",True),("flexure","Flexure",True),("shear_major","Shear Major",False),("shear","Shear",False)]:
                bundle=result.get(key)
                if bundle is not None:
                    st.subheader(title); x=bundle_df(bundle,moment=moment)
                    if not x.empty: st.dataframe(x, hide_index=True, use_container_width=True)
                    for note in getattr(bundle,"notes",[]): st.info(note)
        with tabs[1]:
            audit=result.get("audit_report")
            if audit is not None and audit.items:
                st.dataframe(pd.DataFrame(audit.as_dicts()), hide_index=True, use_container_width=True)
                if audit.has_blockers: st.error("Blocking exact-data branches detected.")
                elif audit.has_warnings: st.warning("Warnings detected for approximate or geometric idealized logic.")
            else:
                st.success("No audit flags detected.")
        with tabs[2]:
            st.json(result.get("geometry_source",{}))
        with tabs[3]:
            rows=[{"Family":i.family,"Chapter":i.chapter,"Description":i.description,"Required Inputs":", ".join(i.required_inputs)} for i in normative_gap_registry_v2()]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if section is not None and result is not None and not err:
        st.subheader("P-M Diagram")
        pc=result["compression"].controlling.design_strength if result.get("compression") and result["compression"].controlling else 0.0
        mc=result["flexure_major"].controlling.design_strength if result.get("flexure_major") and result["flexure_major"].controlling else (result["flexure"].controlling.design_strength if result.get("flexure") and result["flexure"].controlling else 0.0)
        fig=go.Figure()
        pc_kn=n_to_kn(pc); mc_knm=nmm_to_knm(mc)
        fig.add_scatter(x=[0,mc_knm*0.4,mc_knm*0.8,mc_knm,mc_knm*0.8,mc_knm*0.4,0], y=[pc_kn,pc_kn*0.85,pc_kn*0.55,0,-pc_kn*0.55,-pc_kn*0.85,-pc_kn], mode="lines", name="Capacity")
        fig.add_scatter(x=[nmm_to_knm(vals["Mux"])], y=[n_to_kn(vals["Pu"])], mode="markers", marker_size=10, name="Demand")
        fig.update_layout(height=320, margin=dict(l=10,r=10,t=10,b=10), xaxis_title="M (kN·m)", yaxis_title="P (kN)")
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Properties")
        st.dataframe(props_df(section), hide_index=True, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
