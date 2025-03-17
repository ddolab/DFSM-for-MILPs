using LinearAlgebra, Random, Gurobi, GAMS, DataFrames, CSV, Printf, BARON, JuMP, Ipopt;
using Distributed
using JuMP, Gurobi, CPUTime

S=700 # number of data points to be generated
T=30 # horizon length


dtt=zeros(S)

z1=zeros(S,T)
Eo1=zeros(S)
Ptdes1=zeros(S,T)
PENG=zeros(S,T)
PBAT=zeros(S,T)
EE=zeros(S,T+1)
global u=zeros(S)
global counter=0
global holder=zeros(T-1)
global obv=zeros(S)
global Emax
global Pmax

swi=3 # the number of switching actions allowed (S).
tou=5

    ## randomly generating the cost vecotr ##
global c=zeros(3*T+1) 

for i=1:T
    c[i+T+1]= 11+rand(-50:50)/500
    c[i+2T+1]=1.5+rand(-50:50)/5000
end

for i=1:T
    c[i]=-rand(10:30)/1000
end

c[T+1]=-rand(25:50)/10

for k=1:S
    tt=time()
    global u
    Ptdes=zeros(T)
    Eo=95              
    global Emax=100 
    global Pmax=1

    u[k]=rand(100:2000)/1000 
    for i=1:T
       Ptdes[i]=(sin(u[k]^0.5*i)+1)/5
    end
    Ptdes1[k,:]=Ptdes
    Eo1[k]=Eo
    Q1= Model(optimizer_with_attributes(Gurobi.Optimizer))
        @variables(Q1, begin
        Ptbat[1:T]
        Pmax>=Pteng[1:T]>=0
        Emax>=E[1:T+1]>=0
        end)
    @variable(Q1, swi>=z[1:T]>=0, Int)
    @constraint(Q1,E[1]==Eo)
            for i=1:T

                @constraint(Q1, E[i+1]==E[i]-tou*Ptbat[i]) 
                @constraint(Q1, Ptbat[i]+Pteng[i]>=Ptdes[i])
                @constraint(Q1, Pteng[i]<=z[i]*Pmax/swi)
                
            end

        @objective(Q1, Min, (-c[T+1]*(Emax-E[T+1])+sum(c[i]*E[i] for i=1:T)+sum((c[i+T+1]*Pteng[i]+c[i+2T+1]*z[i]) for i=1:T)))
        optimize!(Q1)

        if (primal_status(Q1)==NO_SOLUTION)
            global counter=counter+1
            obv[k]=0
            k=k+1
        else
            obv[k]=getobjectivevalue(Q1)
            z1[k,:]=value.(z)
            PENG[k,:]=value.(Pteng)
            PBAT[k,:]=value.(Ptbat)
            EE[k,:]=value.(E)
            println(k)
        
        end
        dtt[k]=time()-tt
       
end

for i=1:S
    for j=1:T
        z1[i,j]=abs(z1[i,j])
      z1[i,j]=round(z1[i,j])
    end
end


MM=unique((i->begin u[i] end),1:S)
global EO1=zeros(Float64, size(MM,1))
global PTDES1=zeros(Float64, size(MM,1), T)
global PENG1=zeros(Float64, size(MM,1),T)
global PBAT1=zeros(Float64, size(MM,1),T)
global EE1=zeros(Float64, size(MM,1),T+1)
global Z1=zeros(Float64, size(MM,1),T)
global OBV=zeros(Float64, size(MM,1))
for i=1:lastindex(MM,1)
    global EO1[i]=Eo1[MM[i]]
    global PTDES1[i,:]=Ptdes1[MM[i],:]
    global PENG1[i,:]=PENG[MM[i],:]
    global PBAT1[i,:]=PBAT[MM[i],:]
    global EE1[i,:]=EE[MM[i],:]
    global Z1[i,:]=z1[MM[i],:]
    global OBV[i]=obv[MM[i]]
end
u=unique(u,dims=1)
round.(PENG1;digits=2)
round.(EE1;digits=2)
round.(PBAT1;digits=2)



global counter1=0
for i=1:size(EO1,1)
    if (EE1[i,1]==0)
        global counter1=counter1+1
    end
end

EO11=zeros((size(EO1,1)-counter1))
PTDES11=zeros((size(EO1,1)-counter1),T)
PENG11=zeros((size(EO1,1)-counter1),T)
PBAT11=zeros((size(EO1,1)-counter1),T)
EE11=zeros((size(EO1,1)-counter1),T+1)
Z11= zeros((size(EO1,1)-counter1),T)
u1=zeros((size(EO1,1)-counter1))

global kk=1
for i=1:size(EO1,1)
    if (EE1[i,1]>0)
        global kk
        EO11[kk]=EO1[i]
        PTDES11[kk,:]=PTDES1[i,:]
        PENG11[kk,:]=PENG1[i,:]
        PBAT11[kk,:]=PBAT1[i,:]
        EE11[kk,:]=EE1[i,:]
        Z11[kk,:]= Z1[i,:]
        u1[kk]=u[i]
        kk=kk+1
    end
end

output_file=open("C:/data_file.jl","w")  ### change destination ###
write(output_file, "Eo1= ") 
show(output_file, EO11) 
write(output_file, "; \n \n")

write(output_file, "Ptdes1= ") 
show(output_file, PTDES11) 
write(output_file, "; \n \n")

write(output_file, "Peng1= ") 
show(output_file, abs.(round.(PENG11;digits=2))) 
write(output_file, "; \n \n")

write(output_file, "Pbat1= ") 
show(output_file, (round.(PBAT11;digits=2))) 
write(output_file, "; \n \n")

write(output_file, "E1= ") 
show(output_file, abs.(round.(EE11;digits=2))) 
write(output_file, "; \n \n")

write(output_file, "z1= ") 
show(output_file, Z11) 
write(output_file, "; \n \n")

write(output_file, "u= ") 
show(output_file, u1) 
write(output_file, "; \n \n")

write(output_file, "swi= ") 
show(output_file, swi) 
write(output_file, "; \n \n")

write(output_file, "cc= ") 
show(output_file, c) 
write(output_file, "; \n \n")

write(output_file, "tim1= ") 
show(output_file, dtt) 
write(output_file, "; \n \n")

close(output_file)