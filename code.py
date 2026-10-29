"""
WEEK 4: Performance Metrics Analysis and Dashboard Design
Run: pip install numpy pandas matplotlib  then  python week4_code.py
PROCESS
1. Simulate 6,000 shipments (seed 42): region, carrier, mode, distance, weight, planned/actual days, cost, damage.
2. Simulate monthly COGS and average inventory (for inventory turnover).
3. Calculate KPIs: On-time %, Cost per shipment, Inventory turnover, Damage %, Transit time, Cost per km.
4. Group by month, carrier, region, mode. Compare Q4 with Jan-Sep.
5. Draw charts, wireframe and dashboard mock-up.
OUTPUT: output/week4_shipments_dataset.csv, output/results_week4.json, output/img/w4_*.png
"""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
os.makedirs('output/img',exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
NAVY='#1F3A5F'; TEAL='#2A9D8F'; ORG='#E76F51'; GOLD='#E9C46A'; GREY='#8D99AE'
R={}
# ================= WEEK 4 =================
rng=np.random.default_rng(42)
N=6000
dates=pd.to_datetime('2025-01-01')+pd.to_timedelta(rng.integers(0,365,N),unit='D')
region=rng.choice(['North','South','East','West','Central'],N,p=[.2,.25,.2,.2,.15])
carrier=rng.choice(['SwiftHaul','BlueLine','RoadKing','AeroFast'],N,p=[.35,.3,.25,.1])
mode={'SwiftHaul':'Road','BlueLine':'Rail','RoadKing':'Road','AeroFast':'Air'}
df=pd.DataFrame({'shipment_id':[f'SH{i:05d}' for i in range(1,N+1)],'ship_date':dates,'region':region,'carrier':carrier})
df['mode']=df.carrier.map(mode)
df['distance_km']=np.round(rng.gamma(4,150,N)+80).astype(int)
df['weight_kg']=np.round(rng.gamma(3,120,N)+20).astype(int)
m=df.ship_date.dt.month
season=np.where(m.isin([10,11,12]),0.07,np.where(m.isin([6,7,8]),0.03,0))  # peak delay risk
base_late={'SwiftHaul':.17,'BlueLine':.22,'RoadKing':.26,'AeroFast':.08}
p_late=df.carrier.map(base_late)+season+np.where(df.region=='West',0.04,0)
df['planned_days']=np.maximum(1,np.round(df.distance_km/np.where(df['mode']=='Air',900,np.where(df['mode']=='Rail',450,380)))+1).astype(int)
late=rng.random(N)<p_late
df['actual_days']=df.planned_days+np.where(late,rng.integers(1,4,N),0)
rate={'Road':0.9,'Rail':0.6,'Air':3.2}
df['cost_inr']=np.round(df['mode'].map(rate)*df.distance_km*(1+df.weight_kg/800)+rng.normal(250,40,N)+np.where(late,rng.normal(150,40,N),0)).astype(int)
df['on_time']=(df.actual_days<=df.planned_days).astype(int)
df['damaged']=(rng.random(N)<np.where(df['mode']=='Road',.022,np.where(df['mode']=='Rail',.012,.008))).astype(int)
df['month']=df.ship_date.dt.to_period('M').astype(str)
df.sort_values('ship_date').to_csv('output/week4_shipments_dataset.csv',index=False)
# Inventory: monthly COGS and avg inventory
cogs=np.array([4.2,4.0,4.3,4.4,4.6,4.7,4.9,5.0,5.2,6.0,6.8,7.2])*1e6*1.0
avg_inv=np.array([2.1,2.1,2.15,2.2,2.2,2.3,2.35,2.4,2.5,2.7,2.9,2.8])*1e6
months=sorted(df.month.unique())
mon=df.groupby('month').agg(ships=('shipment_id','count'),otd=('on_time','mean'),cost=('cost_inr','mean'),dmg=('damaged','mean'),days=('actual_days','mean')).reset_index()
mon['turnover']=cogs/avg_inv
mon['dmg']*=100; mon['otd']*=100
R['w4']={'n':N,'otd':df.on_time.mean()*100,'cost':df.cost_inr.mean(),'dmg':df.damaged.mean()*100,'transit':df.actual_days.mean(),
 'turn_annual':cogs.sum()/avg_inv.mean(),'turn_month_avg':float((cogs/avg_inv).mean()),
 'cost_median':float(df.cost_inr.median()),'cost_std':float(df.cost_inr.std()),
 'otd_min':[mon.month[mon.otd.idxmin()],mon.otd.min()],'otd_max':[mon.month[mon.otd.idxmax()],mon.otd.max()],
 'cost_cpkm':float(df.cost_inr.sum()/df.distance_km.sum()),
 'by_carrier':df.groupby('carrier').agg(n=('on_time','size'),otd=('on_time','mean'),cost=('cost_inr','mean'),dmg=('damaged','mean')).round(4).reset_index().to_dict('records'),
 'by_region':df.groupby('region').agg(n=('on_time','size'),otd=('on_time','mean'),cost=('cost_inr','mean')).round(4).reset_index().to_dict('records'),
 'by_mode':df.groupby('mode').agg(n=('on_time','size'),otd=('on_time','mean'),cost=('cost_inr','mean')).round(4).reset_index().to_dict('records'),
 'monthly':mon.round(2).to_dict('records'),
 'q4_otd':float(df[m.isin([10,11,12])].on_time.mean()*100),'q123_otd':float(df[~m.isin([10,11,12])].on_time.mean()*100),
 'otd_ci':float(1.96*np.sqrt(df.on_time.mean()*(1-df.on_time.mean())/N)*100),
 'corr_dist_cost':float(df.distance_km.corr(df.cost_inr)),
 'late_cost_extra':float(df[df.on_time==0].cost_inr.mean()-df[df.on_time==1].cost_inr.mean())}
mons=[pd.Period(x).strftime('%b') for x in mon.month]
fig,ax=plt.subplots(figsize=(7,3.2)); ax.plot(mons,mon.otd,marker='o',color=NAVY); ax.axhline(85,ls='--',color=ORG,label='Target 85%'); ax.set_ylabel('On-time delivery (%)'); ax.set_title('Monthly On-Time Delivery Rate, 2025'); ax.legend(); ax.set_ylim(60,95); plt.tight_layout(); plt.savefig('output/img/w4_otd.png'); plt.close()
fig,ax=plt.subplots(figsize=(7,3.2)); ax.bar(mons,mon.cost,color=TEAL); ax.set_ylabel('Avg cost per shipment (INR)'); ax.set_title('Average Cost per Shipment by Month'); plt.tight_layout(); plt.savefig('output/img/w4_cost.png'); plt.close()
fig,ax=plt.subplots(figsize=(7,3.2)); ax.plot(mons,mon.turnover,marker='s',color=ORG); ax.set_ylabel('Turnover (x per month)'); ax.set_title('Monthly Inventory Turnover Ratio (COGS / Avg Inventory)'); plt.tight_layout(); plt.savefig('output/img/w4_turn.png'); plt.close()
bc=df.groupby('carrier').on_time.mean()*100
fig,axs=plt.subplots(1,2,figsize=(7.5,3.2)); axs[0].bar(bc.index,bc.values,color=[NAVY,TEAL,ORG,GOLD]); axs[0].axhline(85,ls='--',color='k',lw=.8); axs[0].set_title('On-time % by Carrier'); axs[0].set_ylim(60,100); axs[0].tick_params(axis='x',rotation=20)
br=df.groupby('region').on_time.mean()*100; axs[1].bar(br.index,br.values,color=GREY); axs[1].axhline(85,ls='--',color='k',lw=.8); axs[1].set_title('On-time % by Region'); axs[1].set_ylim(60,100)
plt.tight_layout(); plt.savefig('output/img/w4_seg.png'); plt.close()
# Dashboard mockup
fig=plt.figure(figsize=(10,6.2)); fig.patch.set_facecolor('#F4F6FA')
fig.text(.03,.95,'LOGISTICS PERFORMANCE DASHBOARD',fontsize=14,weight='bold',color=NAVY)
fig.text(.03,.915,'Filters:  [Year 2025 v]   [Region: All v]   [Carrier: All v]   [Mode: All v]',fontsize=8.5,color=GREY)
cards=[('On-Time Delivery',f"{R['w4']['otd']:.1f}%",'Target 85%'),('Cost / Shipment',f"INR {R['w4']['cost']:.0f}",'Budget 1,500'),('Inventory Turnover',f"{R['w4']['turn_annual']:.1f}x",'Annual'),('Damage Rate',f"{R['w4']['dmg']:.1f}%",'Limit 1.5%')]
for i,(t,v,s) in enumerate(cards):
    ax=fig.add_axes([.03+i*.245,.75,.225,.13]); ax.axis('off'); ax.add_patch(FancyBboxPatch((0,0),1,1,boxstyle='round,pad=0.0,rounding_size=0.06',fc='white',ec='#D0D6E2',transform=ax.transAxes))
    ax.text(.08,.72,t,fontsize=8.5,color=GREY); ax.text(.08,.32,v,fontsize=16,weight='bold',color=NAVY); ax.text(.08,.08,s,fontsize=7.5,color=TEAL)
a1=fig.add_axes([.05,.42,.42,.25]); a1.plot(mons,mon.otd,color=NAVY,marker='o',ms=3); a1.axhline(85,ls='--',color=ORG,lw=.8); a1.set_title('Monthly On-Time %  (line)',fontsize=9,loc='left'); a1.tick_params(labelsize=7)
a2=fig.add_axes([.55,.42,.42,.25]); a2.bar(mons,mon.cost,color=TEAL); a2.set_title('Cost per Shipment  (bar)',fontsize=9,loc='left'); a2.tick_params(labelsize=7)
a3=fig.add_axes([.05,.08,.28,.25]); a3.barh(bc.index,bc.values,color=GREY); a3.set_title('On-Time % by Carrier',fontsize=9,loc='left'); a3.tick_params(labelsize=7); a3.set_xlim(60,100)
a4=fig.add_axes([.40,.08,.28,.25]); a4.bar(br.index,br.values,color=GOLD); a4.set_title('On-Time % by Region',fontsize=9,loc='left'); a4.tick_params(labelsize=7); a4.set_ylim(60,100)
a5=fig.add_axes([.75,.08,.22,.25]); a5.axis('off'); a5.add_patch(FancyBboxPatch((0,0),1,1,boxstyle='round,pad=0,rounding_size=0.05',fc='#FFF4E5',ec=ORG,transform=a5.transAxes))
a5.text(.06,.86,'ALERTS',fontsize=9,weight='bold',color=ORG); a5.text(.06,.58,'! Q4 on-time below target\n! West region lags\n! RoadKing carrier < 75%',fontsize=7.5,va='center',linespacing=1.6)
plt.savefig('output/img/w4_dashboard.png'); plt.close()
# Wireframe sketch
fig,ax=plt.subplots(figsize=(8,5)); ax.set_xlim(0,100); ax.set_ylim(0,62); ax.axis('off')
def box(x,y,w,h,t,fc='white'):
    ax.add_patch(plt.Rectangle((x,y),w,h,fc=fc,ec='k',lw=1.2)); ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=8)
box(1,55,98,6,'HEADER: Title  |  Filters (Year, Region, Carrier, Mode)','#E8EDF7')
for i,t in enumerate(['KPI 1\nOn-Time %','KPI 2\nCost / Shipment','KPI 3\nInventory Turnover','KPI 4\nDamage Rate']): box(1+i*24.7,42,23.5,11,t,'#FFF9E5')
box(1,22,48,18,'Trend line: On-Time % by month\n(with target line)','#EAF6F4'); box(51,22,48,18,'Bar chart: Cost per shipment by month','#EAF6F4')
box(1,2,30,18,'Carrier comparison\n(horizontal bars)','#EAF6F4'); box(33,2,30,18,'Region comparison\n(column chart)','#EAF6F4'); box(65,2,34,18,'ALERT PANEL\n(KPIs outside target)','#FDECEA')
ax.set_title('Figure: Hand-drawn style wireframe of dashboard layout',fontsize=9); plt.tight_layout(); plt.savefig('output/img/w4_wireframe.png'); plt.close()


json.dump(R,open('output/results_week4.json','w'),indent=1,default=float)
print('Done. Check the output folder.')