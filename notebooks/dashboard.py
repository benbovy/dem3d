import os
import shutil
import json
import zipfile
from math import ceil, floor
from tqdm.notebook import tqdm
import requests
from ipyleaflet import Map, basemaps
from osgeo import gdal
import numpy as np
import zarr
import xarray as xr
from ipygany import Scene, PolyMesh, IsoColor, Component
from ipyleaflet import DrawControl
from ipywidgets import Play, IntProgress, link, HBox, VBox
import matplotlib.pyplot as plt
from IPython.display import display

import xsimlab as xs
from fastscape.models import basic_model
from fastscape.processes import SurfaceTopography



NNODES = 6000


@xs.process
class InitDEM:
    dem = xs.variable(dims=('y', 'x'))
    elevation = xs.foreign(SurfaceTopography, 'elevation', intent='out')
    
    def initialize(self):
        self.elevation = self.dem
        

model = basic_model.update_processes({'init_topography': InitDEM})


def create_setup(dem):
    
    resolution = 20
    grid_shape = [dem.sizes['y'], dem.sizes['x']]
    grid_length = list(np.array(grid_shape) * 40)

    in_ds = xs.create_setup(
        model=model,
        clocks={
            'time': np.arange(0, 1e6 + 1e4, 2e4),
            'out': np.arange(0, 1e6 + 1e4, 2e4),
        },
        master_clock='time',
        input_vars={
            'grid__shape': grid_shape,
            'grid__length': grid_length,
            'boundary__status': 'fixed_value',
            'init_topography__dem': dem.values,
            'uplift__rate': 1e-6,
            'spl': {
                'k_coef': 2e-6,
                'area_exp': 0.6,
                'slope_exp': 1.5
            },
            'diffusion__diffusivity': 1e-3
        },
        output_vars={
            'topography__elevation': 'out',
        }
    )
    
    return in_ds


class Dashboard:

    def __init__(self, m, notif, dem2d=None, dem3d=None):
        self.map = m
        self.notif = notif
        self.dem2d = dem2d
        self.dem3d = dem3d
        self.tile_dir = 'dem_tiles'

    def start(self):
        self.draw_control = DrawControl()
        self.draw_control.polygon = {}
        self.draw_control.polyline = {}
        self.draw_control.circlemarker = {}
        self.draw_control.rectangle = {
            'shapeOptions': {
                'fillOpacity': 0.5
            }
        }
        self.draw_control.on_draw(self.run_model)
        self.map.add_control(self.draw_control)
    
    def get_dem(self, *args, **kwargs):
        lonlat = self.draw_control.last_draw['geometry']['coordinates'][0]
        lats = [ll[1] for ll in lonlat]
        lons = [ll[0] for ll in lonlat]
        lt0, lt1 = min(lats), max(lats)
        ln0, ln1 = min(lons), max(lons)
        os.makedirs(self.tile_dir, exist_ok=True)
        with open(self.tile_dir + '.json') as f:
            tiles = json.loads(f.read())
        lat = lat0 = floor(lt0 / 5) * 5
        lon = lon0 = floor(ln0 / 5) * 5
        lat1 = ceil(lt1 / 5) * 5
        lon1 = ceil(ln1 / 5) * 5
        ny = int(round((lat1 - lat0) / (5 / NNODES)))
        nx = int(round((lon1 - lon0) / (5 / NNODES)))
        zarr_path = os.path.join(self.tile_dir + '.zarr')
        if os.path.exists(zarr_path):
            shutil.rmtree(zarr_path)
        z = zarr.open(zarr_path, mode='w', shape=(ny, nx), chunks=(NNODES, NNODES), dtype='float32')
        done = False
        while not done:
            self.make_zarr(lat, lon, tiles, lat0, lat1, lon0, lon1, z)
            lon += 5
            if lon >= ln1:
                lon = lon0
                lat += 5
                if lat >= lt1:
                    done = True
        y = np.arange(lat0, lat1, 5 / NNODES)
        x = np.arange(lon0, lon1, 5 / NNODES)
        
        dem = xr.DataArray(z, coords=[y, x], dims=['y', 'x']).sel(y=slice(lt0, lt1), x=slice(ln0, ln1))
        return dem
    
    def run_model(self, *args, **kwargs):
        dem = self.get_dem(*args, **kwargs)
        in_ds = create_setup(dem)
        with self.notif:
            print("Starting landscape evolution model run...")
            with xs.monitoring.ProgressBar():
                out_ds = in_ds.xsimlab.run(model=model)
        
        out_ds['x'] = dem.x
        out_ds['y'] = dem.y

        #print(dem)
        #print(out_ds.topography__elevation.isel(out=-1))
        self.show_dem3d(dem, out_ds)

    def show_dem(self, *args, **kwargs):
        dem = self.get_dem(*args, **kwargs)
        self.show_dem2d(dem)
        self.show_dem3d(dem)

    def show_dem2d(self, dem):
        fig = dem.plot.imshow()
        with self.dem2d:
            display(fig)
            plt.show()
    
    def get_mesh_elements(self, dem, return_triangles=True):
        nr = len(dem.y)
        nc = len(dem.x)
        
        if return_triangles:

            triangle_indices = np.empty((nr - 1, nc - 1, 2, 3), dtype=int)

            r = np.arange(nr * nc).reshape(nr, nc)

            triangle_indices[:, :, 0, 0] = r[:-1, :-1]
            triangle_indices[:, :, 1, 0] = r[:-1, 1:]
            triangle_indices[:, :, 0, 1] = r[:-1, 1:]

            triangle_indices[:, :, 1, 1] = r[1:, 1:]
            triangle_indices[:, :, :, 2] = r[1:, :-1, None]

            triangle_indices.shape = (-1, 3)

            xx, yy = np.meshgrid(dem.x, dem.y, sparse=True)
        
        else:
            
            triangles_indices = None

        vertices = np.empty((nr, nc, 3))
        vertices[:, :, 0] = xx * np.pi / 180
        vertices[:, :, 1] = yy * np.pi / 180
        vertices[:, :, 2] = dem.values

        lon = np.copy(vertices[:, :, 0])
        lat = np.copy(vertices[:, :, 1])
        alt = np.copy(vertices[:, :, 2])
        r = 6371e3  # Earth's radius in meters
        f = (alt + r) / r * 10  # normalized factor
        vertices[:, :, 0] = np.sin(np.pi / 2 - lat) * np.cos(lon) * f
        vertices[:, :, 1] = np.sin(np.pi / 2 - lat) * np.sin(lon) * f
        vertices[:, :, 2] = np.cos(np.pi / 2 - lat) * f

        vertices = vertices.reshape(nr * nc, 3)
        
        return triangle_indices, vertices

    def show_dem3d(self, dem, out_ds):
        triangle_indices, vertices = self.get_mesh_elements(dem)
        
        height_component = Component(name='value', array=dem.values)

        mesh = PolyMesh(
            vertices=vertices,
            triangle_indices=triangle_indices,
            data={'height': [height_component]}
        )

        colored_mesh = IsoColor(mesh, input=('height', 'value'), min=np.min(dem.values), max=np.max(dem.values))
        
        def load_step(change):
            new_dem = out_ds.topography__elevation.isel(out=change['new'])
            _, vertices = self.get_mesh_elements(new_dem)
            mesh.vertices = vertices
            height_component.array = new_dem.values
        
        play = Play(description='Step:', min=1, max=50, value=1)
        play.observe(load_step, names=['value'])

        progress = IntProgress(value=1, step=1, min=1, max=50)
        link((progress, 'value'), (play, 'value'))

        stepper = HBox((play, progress))
        
        scene = Scene([colored_mesh])

        with self.dem3d:
            display(VBox((stepper, scene)))

    def make_zarr(self, lat, lon, tiles, lat0, lat1, lon0, lon1, z):
        if lat < 0:
            fname = 's'
        else:
            fname = 'n'
        fname += str(abs(lat)).zfill(2)
        if lon < 0:
            fname += 'w'
        else:
            fname += 'e'
        fname += str(abs(lon)).zfill(3)
        fname += '_con_grid.zip'
        url = ''
        for continent in tiles:
            if fname in tiles[continent][1]:
                url = tiles[continent][0] + fname
                break
        if url:
            filename = os.path.basename(url)
            name = filename[:filename.find('_grid')]
            adffile = os.path.join(self.tile_dir, name, name, 'w001001.adf')
            zipfname = os.path.join(self.tile_dir, filename)

            with self.notif:
                if os.path.exists(adffile):
                    print('Already downloaded ' + adffile)
                else:
                    print('Downloading ' + url)
                    r = requests.get(url, stream=True)
                    with open(zipfname, 'wb') as f:
                        total_length = int(r.headers.get('content-length'))
                        for chunk in tqdm(r.iter_content(chunk_size=1024), total=(total_length/1024) + 1):
                            if chunk:
                                f.write(chunk)
                                f.flush()
                    zip = zipfile.ZipFile(zipfname)
                    zip.extractall(self.tile_dir)

            dem = gdal.Open(adffile)
            geo = dem.GetGeoTransform()
            ySize, xSize = dem.RasterYSize, dem.RasterXSize
            dem = dem.ReadAsArray()
            # data is padded into a NNODESxNNODES array (some tiles may be smaller)
            array_5x5 = np.full((NNODES, NNODES), -32768, dtype='int16')
            y0 = int(round((geo[3] - lat - 5) / geo[5]))
            y1 = NNODES - int(round((lat - (geo[3] + geo[5] * ySize)) / geo[5]))
            x0 = int(round((geo[0] - lon) / geo[1]))
            x1 = NNODES - int(round(((lon + 5) - (geo[0] + geo[1] * xSize)) / geo[1]))
            array_5x5[y0:y1, x0:x1] = dem
            array_5x5 = np.where(array_5x5==-32768, np.nan, array_5x5.astype('float32'))
            y0 = z.shape[0] + (lat - lat1) // 5 * NNODES
            y1 = y0 + NNODES
            x0 = (lon - lon0) // 5 * NNODES
            x1 = x0 + NNODES
            z[y0:y1, x0:x1] = array_5x5[::-1]
