clc
clear
close all

DataSize = [ 256, 256, 207 ];
PatchSize = [ 4, 4, 4 ];
PatchStep = [ 1, 1, 1 ];

% % read training images 
MAP1_Path = 'simulatedHRRT_subject04_10_noise_1_GS_beta_0.0015_it10_subset16.i';
Fid = fopen( MAP1_Path, 'rb' );
FileData = fread( Fid, inf, 'float' );
MAP1_Data  = reshape( FileData, DataSize );

MAP2_Path = 'simulatedHRRT_subject04_10_noise_1_GS_beta_0.0025_it10_subset16.i';
Fid = fopen( MAP2_Path, 'rb' );
FileData = fread( Fid, inf, 'float' );
MAP2_Data  = reshape( FileData, DataSize );

MAP3_Path = 'simulatedHRRT_subject04_10_noise_1_GS_beta_0.0025_it10_subset16.i';
Fid = fopen( MAP3_Path, 'rb' );
FileData = fread( Fid, inf, 'float' );
MAP3_Data  = reshape( FileData, DataSize );

Phantom_Path = 'subject04_act_hrrt_256_256_207.rawb';
Fid = fopen( Phantom_Path, 'rb' );
FileData = fread( Fid, inf, 'float' );
Phantom_Data  = reshape( FileData, DataSize );

fclose( 'all' );

%% Generate training and validating label 
% calculate maximum intensity
max_train = max( MAP1_Data(:) );                                            

% extract the effective region
Phantom_Data  = Phantom_Data( 44:214, 35:230, 20:194 ) ./ max_train; 

% extract 3D patches
[ PatData, PatNum ] = PatchExtract( Phantom_Data, PatchSize, PatchStep );

% calculate variance of each patch
varValue=[];                                                               
for i=1:168
    varValue_batch=var( PatData( :, (1+(i-1)*PatNum/168):i*PatNum/168 ) );
    varValue=[varValue varValue_batch];
end

% sort variances in decreasing order
[varV, Idex]=sort(varValue,'descend'); 

% select 200000 training patches
num = randperm( PatNum-100000 )+100000;
Idex_m = Idex( num(1:100000) );
TraL = PatData( :, [Idex( 1 : 100000 ), Idex_m] );
% shuffle patches
IIdex = randperm( 200000 );
TraL = TraL( :, IIdex );  

% select 10000 validating patches
Idex_n = Idex( num(100001:110000) );
ValL = PatData( :, Idex_n );

% removd DC of patches
TraLab = remove_dc( TraL, 'columns' );
ValLab = remove_dc( ValL, 'columns' );

clear PatData 

%% Generate training and validating data 
% MAP 1
MAP1_Data  = MAP1_Data( 44:214, 35:230, 20 : 194 ) ./ max_train;            

[ PatData, PatNum ] = PatchExtract( MAP1_Data, PatchSize, PatchStep );

TraData1 = PatData( :, [Idex( 1 : 100000 ), Idex_m] );
TraData1 = remove_dc( TraData1, 'columns' );

ValData1 = PatData( :, Idex_n );
ValData1 = remove_dc( ValData1, 'columns' );
clear PatData 

% MAP 2
MAP2_Data  = MAP2_Data( 44:214, 35:230, 20 : 194 ) ./ max_train;            

[ PatData, PatNum ] = PatchExtract( MAP2_Data, PatchSize, PatchStep );

TraData2 = PatData( :, [Idex( 1 : 100000 ),Idex_m] );
TraData2 = remove_dc( TraData2, 'columns' );

ValData2 = PatData( :, Idex_n );
ValData2 = remove_dc( ValData2, 'columns' );
clear PatData 

% MAP 3
MAP3_Data  = MAP3_Data( 44:214, 35:230, 20 : 194 ) ./ max_train;            

[ PatData, PatNum ] = PatchExtract( MAP3_Data, PatchSize, PatchStep );

TraData3 = PatData( :, [Idex( 1 : 100000 ),Idex_m]);
TraData3 = remove_dc( TraData3, 'columns' );

ValData3 = PatData( :, Idex_n );
ValData3 = remove_dc( ValData3, 'columns' );
clear PatData 

% combine 3 patches
Tra_add=[TraData1; TraData2; TraData3];
Val_add=[ValData1; ValData2; ValData3];

% shuffle training data
TraData_add=Tra_add(:,IIdex);
ValData_add=Val_add;

% normalize training and validating label
[TraLab,settings1] = mapminmax(TraLab);
ValLab=mapminmax.apply(ValLab,settings1);

% normalize training and validating data
[TraData_add,settings2] = mapminmax(TraData_add);
ValData_add=mapminmax.apply(ValData_add,settings2);

save train_validate_data_label TraData_add ValData_add TraLab ValLab settings1 settings2
