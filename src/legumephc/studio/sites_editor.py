"""Atomic, multi-site geometry editor."""
from copy import deepcopy
import tkinter as tk
from tkinter import ttk
import numpy as np
from .geometry_editor import SHAPE_CHOICES, canonical_shape, shape_choice, safe_number, editor_value, epsilon_from_editor
from ..motifs import triangular_motifs


def number_text(value):
    return format(float(value), '.12g')


class SitesDialog(tk.Toplevel):
    def __init__(self, parent, *, motifs, representation, lattice, center, scale=1.0, add=False):
        super().__init__(parent)
        self.title('Edit sites')
        self.transient(parent)
        self.result = None
        self.representation, self.lattice, self.center, self.scale = representation, lattice, np.asarray(center), scale
        self.rows = []
        ttk.Label(self, text='Geometry parameters before affine transformation').pack(anchor='w', padx=10, pady=8)
        self.table = ttk.Frame(self, padding=10)
        self.table.pack(fill='both', expand=True)
        self.positions = ttk.Frame(self, padding=10)
        self.position_visible = False
        bar = ttk.Frame(self, padding=10)
        bar.pack(fill='x')
        ttk.Button(bar, text='Add site', command=self.add_site).pack(side='left')
        ttk.Button(bar, text='Position ▸', command=self.toggle_positions).pack(side='left')
        ttk.Button(bar, text='Apply', command=self.apply).pack(side='right')
        ttk.Button(bar, text='Cancel', command=self.destroy).pack(side='right')
        self.error = ttk.Label(self, foreground='#b00020')
        self.error.pack(fill='x', padx=10, pady=5)
        for motif in motifs:
            self.append_row(motif)
        if add:
            self.add_site()
        self.bind('<Escape>', lambda event: self.destroy())
        self.grab_set()

    def append_row(self, motif):
        source = deepcopy(motif)
        shape = shape_choice(source['kind'], source.get('sides'))
        name = source.get('name', '')
        values = {'name':name, 'shape':shape, 'radius':number_text(source['radius']), 'angle':number_text(source.get('angle_degrees',0)), 'material':number_text(editor_value(source['epsilon'], self.representation)), 'sides':str(source.get('sides', 6)), 'x': '' if source['center'][0] is None else number_text(source['center'][0]), 'y': '' if source['center'][1] is None else number_text(source['center'][1])}
        row = {'source':source, 'initial':dict(values), 'vars':{key:tk.StringVar(self, value) for key,value in values.items()}, 'auto_name':not name or name.replace(' ', '').lower().endswith(shape.replace(' ', '').lower())}
        self.rows.append(row)
        self.render()

    def render(self):
        for frame in (self.table, self.positions):
            for child in frame.winfo_children():
                child.destroy()
        keys = ('name','shape','radius','angle','material','sides')
        for col,label in enumerate(('Name','Shape','Radius r/a','Rotation deg','n' if self.representation=='n' else 'ε','Sides')):
            ttk.Label(self.table,text=label).grid(row=0,column=col)
        for index,row in enumerate(self.rows):
            row['widgets'] = {}
            for col,key in enumerate(keys):
                var = row['vars'][key]
                widget = ttk.Combobox(self.table,textvariable=var,values=SHAPE_CHOICES,state='readonly',width=17) if key=='shape' else ttk.Entry(self.table,textvariable=var,width=15 if key=='name' else 11)
                widget.grid(row=index+1,column=col,padx=3,pady=3)
                row['widgets'][key]=widget
                if key=='shape':
                    widget.bind('<<ComboboxSelected>>',lambda event,r=row:self.shape_changed(r))
            ttk.Button(self.table,text='Remove',command=lambda r=row:self.remove(r)).grid(row=index+1,column=6)
            ttk.Label(self.positions,text=f'Site {index+1} position x/a, y/a').grid(row=index,column=0)
            for col,key in enumerate(('x','y'),1):
                widget=ttk.Entry(self.positions,textvariable=row['vars'][key],width=15)
                widget.grid(row=index,column=col)
                row['widgets'][key]=widget
            self.shape_changed(row, rename=False)

    def shape_changed(self,row,rename=True):
        shape=row['vars']['shape'].get()
        row['widgets']['angle'].configure(state='disabled' if shape=='Circle' else 'normal')
        row['widgets']['sides'].configure(state='normal' if shape=='Regular polygon' else 'disabled')
        if rename:
            old_name=row['vars']['name'].get()
            if row['auto_name'] and old_name in (row['initial']['name'],row.get('generated','')):
                row['generated']=shape.replace(' ','')
                row['vars']['name'].set(row['generated'])
            if shape in ('Triangle','Square'):
                row['vars']['sides'].set('3' if shape=='Triangle' else '4')
            elif shape=='Regular polygon':
                try:
                    valid=safe_number(row['vars']['sides'].get())>=3
                except ValueError:
                    valid=False
                if not valid:
                    row['vars']['sides'].set('6')

    def toggle_positions(self):
        self.position_visible=not self.position_visible
        if self.position_visible:
            self.positions.pack(fill='x')
        else:
            self.positions.pack_forget()

    def add_site(self):
        motif={'name':'','kind':'circle','radius':.2,'angle_degrees':0,'epsilon':1.,'sides':0,'center':[None,None]}
        if self.lattice=='triangular' and len(self.rows)==1:
            row=self.rows[0]
            try:
                current=np.array([safe_number(row['vars'][k].get()) for k in ('x','y')])
                if np.allclose(current,self.center,rtol=0,atol=1e-12):
                    pair=triangular_motifs((.2,.2),scale=self.scale)
                    for key,value in zip(('x','y'),pair[0]['center']):
                        row['vars'][key].set(number_text(value))
                    motif['center']=pair[1]['center']
            except ValueError:
                pass
        self.append_row(motif)
        if motif['center'][0] is None and not self.position_visible:
            self.toggle_positions()

    def remove(self,row):
        if len(self.rows)==1:
            self.error.configure(text='Keep at least one site')
            return
        self.rows.remove(row)
        self.render()

    def apply(self):
        result=[]
        for index,row in enumerate(self.rows):
            field='shape'
            try:
                v=row['vars']
                field='sides' if v['shape'].get()=='Regular polygon' else 'shape'
                if field=='sides' and not safe_number(v['sides'].get()).is_integer():
                    raise ValueError('polygon sides must be an integer')
                kind,sides=canonical_shape(v['shape'].get(),v['sides'].get())
                def number(key):
                    nonlocal field
                    field=key
                    if ',' in v[key].get():
                        raise ValueError('enter one number in this row; use Add site for another site')
                    return safe_number(v[key].get())
                radius=number('radius')
                if radius<=0:
                    raise ValueError('radius must be positive')
                angle=0. if kind=='circle' else number('angle')
                center=[number('x'),number('y')]
                material=number('material')
                epsilon=epsilon_from_editor(material,self.representation)
                motif={**row['source'],'name':v['name'].get().strip() or v['shape'].get().replace(' ',''),'kind':kind,'sides':sides,'radius':radius,'angle_degrees':angle,'center':center,'epsilon':epsilon}
                # Untouched rounded controls retain the original exact values.
                for key,target in (('radius','radius'),('angle','angle_degrees'),('material','epsilon')):
                    if v[key].get()==row['initial'][key] and (key!='angle' or kind==row['source']['kind']):
                        motif[target]=row['source'].get(target,motif[target])
                for i,key in enumerate(('x','y')):
                    if v[key].get()==row['initial'][key]:
                        motif['center'][i]=row['source']['center'][i]
                result.append(motif)
            except (ValueError,TypeError) as exc:
                self.error.configure(text=f'Site {index+1}, {field}: {exc}')
                if field in ('x','y') and not self.position_visible:
                    self.toggle_positions()
                row['widgets'].get(field,row['widgets']['radius']).focus_set()
                return
        self.result=result
        self.destroy()
