%% 
%% Adapted from: Kim-Chuan Toh and Yinyu Ye
%% By Konovalov Kirill
%%*************************************************************************
%%
%%  
%%
%%
%% the programm iterates over a range of noise factors "nf"
%% and contact map cutoffs 11 + "r" to produce a matrix of RMSDs 
%% obtained after the recovered structure is superimposed to the original
%%
%% contour.ipynb prvides a programm to visualize the matrix
 
   close all;    
   clear;
   for nf = 0:20
       for r = 0:14

           Step = 16/14.0;
           Radius = 11 + r*Step;
       
           noisetype = 'additive';  
           randstate = 1; 
        %%
           rand('state',randstate); 


              P0 = []; 
              protein = "test";
              filename = '1R9H.pdb';
              Porig = readPDB(filename); %% atom positions
              [dim, N] = size(Porig);
              center = Porig*ones(N,1)/N; 
              PP = Porig - center*ones(1,N); 
              BoxScale = 2*ceil(max(max(abs(PP))));

           nfix = size(P0,2);
           [dim,npts] = size(PP); 
        %%
        %% main 
        %%
          
        %%filename
           OPTIONS.alpha       = 1; %% regularization parameter
           OPTIONS.refinemaxit = 1000; 
           OPTIONS.plotyes     = 0; 
           OPTIONS.printyes    = 1; 
           OPTIONS.PP          = PP;   
           OPTIONS.BoxScale    = BoxScale; 
           OPTIONS.nf          = nf; 
        %%
           DD = randistance(P0,PP,Radius,nf,noisetype,randstate);
           [Xopt,Yopt] = SNLsolver(P0,DD,dim,OPTIONS);
           if (Xopt == 0) 
               MSES(nf+1,r+1) = 999;
               continue;
           end
           
           tvar = max(0,diag(Yopt)'-sum(Xopt.*Xopt));
           Xtmp = matchposition(Xopt,PP,tvar);
  
           errtrue = sum((Xtmp-PP).*(Xtmp-PP));  
           RMSD = sqrt(sum(errtrue))/sqrt(npts); 

           MSES(nf+1,r+1) = RMSD;
           
           name = sprintf('%snormal%drange%.1f.xyz', protein, nf, Radius);
           fileID = fopen(name,'w');
           fprintf(fileID,'93\n\n');
           fprintf(fileID,'C %6.3f\t%6.3f\t%6.3f\n', Xtmp);
           fclose(fileID);
       end
   end
 dlmwrite("matrix.dat",MSES)
   
