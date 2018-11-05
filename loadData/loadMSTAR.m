 function [ trainData,trainLabel,trainAangle] = loadMSTAR( )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%��ȡD:\MSTAR_DATASET�µ�ѵ���Ͳ����ļ������ص�imagedb��ȥ
disp('ver1 loadmstar')
dbnum=0;
%centX=64;
%centY=64;
%��ȡD:\MSTAR_DATASET\17DEG�µ������ļ�������ѵ��
filelist=scanDir( 'D:\MSTAR_DATASET\17DEG');   
num=length(filelist);
trainData=single(zeros([num,64,64]));
trainLabel=uint8(zeros([num,1]));
trainAangle=uint16(zeros([num,1]));
for i=1:num
    
    %filename=filelist(i);����ȡ�����ľ���cell����
    filename=filelist{i};
    [pathstr,name,ext]=fileparts(filename); 
     %��ȡ�ļ���׺���ĺ�3λ����label��ֵ
    if strcmp(ext,'.015')
       label=1;
    elseif strcmp(ext,'.002')
       label=2;
    elseif strcmp(ext,'.004')
       label=3;
    elseif strcmp(ext,'.003')
       label=4;
    elseif strcmp(ext,'.000')
       label=5;
    elseif strcmp(ext,'.001')
       label=6;
    elseif strcmp(ext,'.005')
       label=7;
    elseif strcmp(ext,'.016')
       label=8;
    elseif strcmp(ext,'.025')
       label=9;
    elseif strcmp(ext,'.026')
       label=10;
    else 
       label=0;
    end
    if label==0
        continue;
    end
    [imgdata,header]=readmstar(filename);
    [aangle,dangle]=FindAngle(header);
    aangle=uint16(aangle);
    if mod(aangle,10)==0
       continue;
    end
    %��ͼ����вü�
    [rows,cols]=size(imgdata);
    centX=floor(rows/2);
    centY=floor(cols/2); 
    clipimage=imgdata(centX-32:centX+31,centY-32:centY+31);
    dbnum = dbnum+1;
    rawimage=abs(clipimage);
    rawimage(rawimage<0.01)=0.01;
    trainData(dbnum,:,:)=single(log(rawimage));
    trainLabel(dbnum)=uint8(label);
    trainAangle(dbnum)=aangle;    
end
disp (dbnum)
trainData=trainData(1:dbnum,:,:);
trainLabel=trainLabel(1:dbnum);
trainAangle=trainAangle(1:dbnum);
end


function [Aangle,Dangle]=FindAngle(header)
find=0;
Aangle=-1;
Dangle=-1;
[header_row,header_col]=size(header);
for i=1:header_row
    str = header(i,:);
    str = strtrim(str);
    [name,value]=strtok(str,'=');
    if strcmpi(name,'TargetAz')
        Aangle=str2double(value(2:end));
        find=find+1;
    end
    if strcmpi(name,'DesiredDepression')
        Dangle=str2double(value(2:end));
        find=find+1;
    end
    if find>=2
        break;
    end
end
end


