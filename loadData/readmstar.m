function [imgdata,header]=readmstar(filename)

%************************************************************************
%*                   PHOENIX HEADER PROCESSING SECTION                  *
%************************************************************************

% Initialize some things...
% C:\Users\liusu\Documents\我的工作\SAR
% filename='C:\Users\liusu\Documents\我的工作\SAR Images\MSTAR\15_DEG\COL2\SCENE1\A04\HB14932.017';
%filename='C:\Program Files\Tencent\QQ\Users\732130013\FileRecv\SN_C71\HB04967.004';
fid = 1;
tp = [];
    
for i = 1:1
    header(i,:) = blanks(100);  % Array to hold Phoenix header..
end

i=0;
 
% Open scene for reading...
fid = fopen(filename,'r');

%* Read Phoenix header..extract parameters.. 
while (strcmp(tp,'[EndofPhoenixHeader]') == 0)
  % Get one header string from file...
  z1 = [];
  z1 = fgets(fid); 

  % Scan header string into temp variable tp...
  tp = sscanf(z1,'%s'); 

  % Load header string into header matrix (header)...if non-empty
  if(isempty(tp) == 0)
    i = i+1;
    header(i,:) = zeros(1,100);
    header(i,1:(size(tp,2))) = tp;  
  end
end
fclose(fid);

% Calculate HEADER SIZE (in bytes)...
hdr_size_field = 'PhoenixHeaderLength=';
hdr_size_flag = 0;
i = 0;

while(hdr_size_flag == 0)
  i = i+1;
  hdr_size_flag = strcmp(header(i,1:size(hdr_size_field,2)),hdr_size_field);
end  

hdrsize = str2num(header(i,size(hdr_size_field,2)+1:size(header,2))); 

% Extract NUMBER OF COLUMNS.... 
numcol_field = 'NumberOfColumns=';
numcol_flag = 0;
i = 0;
  
while(numcol_flag == 0)
  i = i+1;
  numcol_flag = strcmp(header(i,1:size(numcol_field,2)),numcol_field);
end 

numcol = str2num(header(i,size(numcol_field,2)+1:size(header,2)));

% Extract NUMBER OF ROWS.... 
numrow_field = 'NumberOfRows=';
numrow_flag = 0;
i = 0;
  
while(numrow_flag == 0)
  i = i+1;
  numrow_flag = strcmp(header(i,1:size(numrow_field,2)),numrow_field);
end 
 
numrow = str2num(header(i,size(numrow_field,2)+1:size(header,2))); 

% Extract SENSOR CALIBRATION FACTOR...
sensor_cal_field = 'SensorCalibrationFactor=';
sensor_cal_flag = 0;
i = 0;
     
while(sensor_cal_flag == 0)
  i = i+1;
  sensor_cal_flag = strcmp(header(i, ...
                                  1:size(sensor_cal_field,2)), ...
                                  sensor_cal_field);
end
 
calfactor = str2num(header(i,size(sensor_cal_field,2)+1:size(header,2))); 


%************************************************************************
%*                        IMAGE PROCESSING SECTION                      *
%************************************************************************

% 2011-11-22*===========================================================
% disp('Processing image data...');
% disp(' ');
% disp([' Num Rows (hgt): ', num2str(numrow)]); 
% disp([' Num Cols (wid): ', num2str(numcol)]); 
% disp(' ');
% disp(' ');
% **************************************************************

fid=fopen(filename,'r','ieee-be');
% Seek to start of clutter scene image data..
fseek(fid,hdrsize,'bof');

% Form normalization scale factor...
% scalefactor = calfactor/65535;
scalefactor = calfactor;

f=fread(fid,'float32');
% Close file..
fclose(fid);

amp=scalefactor*f(1:numcol*numrow);
angle=f(numcol*numrow+1:numcol*numrow*2);
imgdata=amp.*exp(j*angle);
imgdata=reshape(imgdata,numcol,numrow);
imgdata=imgdata.';

%%%%%%%
%imgdata=wiener2(imgdata);


% rawdata=ifty(iftx(imgdata));
% 
% tayrow=taywin(numrow,-35,5);
% 
% taycol=taywin(numcol,-35,5);
% taycol=taycol.';
% tay2=tayrow*taycol;
% 
% rawdata=rawdata./tay2;




%************************************************************************
%*                            DISPLAY SECTION                           *
%************************************************************************

% Put up output display window...

%figure;

% Set up a colortable..use default gray map
%colormap(gray(256));

% Display log10(val+1) scaled image..
% imagesc(log10(abs(imgdata(:,:)/65535) + 1))
 
% Set AXIS Parameters...
% axis image;        % Retain wid to hgt image aspect..
% axis off;          % Turn off axis labelling..

% Brighten image...
% brighten(0.6);%%%

% Last line of vw_clut.m
