clc;
clear all;
close all;

info = nii_read_header();
Volume = nii_read_volume(info);


% filename = 'VSD.Brain.XX.O.MR_Flair.684.mha';
% pathname = ' C:\Users\Ujjwal Baid\Desktop\Patch Based Segmentation';
% %filename = [pathname filename];
% info = mha_read_header(filename, pathname);
% V_FLAIR = mha_read_volume(info);
