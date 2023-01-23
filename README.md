# Sea Ice Classification

This repo contains the code for the Sea Ice Classification Challenge from the 2022-23 cohort of AI4ER at the University of Cambridge.

## Code (& Shared Folder) Structure

This is the current structure of the folders in the GitHub repository and the shared OneDrive folder.

```
Home:.
├───GTC_shared_files
│   ├───EDA
│   ├───FTP_data
│   │   ├───Antarctica_coastline
│   │   ├───clipped_ice_charts
│   │   ├───dual_band_images
│   │   ├───ice_charts
│   │   │   ├───aari_antice_20180104_pl_a
│   │   │   ├───aari_antice_20181220_pl_a
│   │   │   ├───aari_antice_20190314_pl_a
│   │   │   ├───aari_antice_20201105_pl_a
│   │   │   ├───nic_antarc_20180222_pl_a
│   │   │   ├───nic_antarc_20191003_pl_a
│   │   │   ├───nic_antarc_20200117_pl_a
│   │   │   ├───nic_antarc_20200305_pl_a
│   │   │   ├───nic_antarc_20200313_pl_a
│   │   │   ├───nic_antarc_20200917_pl_a
│   │   │   ├───nic_antarc_20211223_pl_a
│   │   │   ├───nis_antarc_20171106_pl_a
│   │   │   ├───nis_antarc_20171223_pl_a
│   │   │   ├───nis_antarc_20180226_pl_a
│   │   │   ├───nis_antarc_20180319_pl_a
│   │   │   ├───nis_antarc_20181203_pl_a
│   │   │   ├───nis_antarc_20181210_pl_a
│   │   │   ├───nis_antarc_20181217_pl_a
│   │   │   └───nis_antarc_20181228_pl_a
│   │   └───rasterised_shapefiles
│   ├───images
│   └───Results
└───sea-ice-classification
```

If you want to create a *.ipynb* in the Home directory of the shared OneDrive folder (**GTC_shared_files**), you would reference the location of data as usual (*i.e.* <code>./FTP_data/dual_band_images/</code>...<code>.tif</code>). On the other hand, if you want to work in the GitHub repository (**sea-ice-classification**), you would have to go up one more level in the folder structure (*i.e.* <code>./../FTP_data/dual_band_images/</code>...<code>.tif</code>).