using LinearAlgebra, Random, Gurobi, GAMS, DataFrames, CSV, Printf, BARON, JuMP, Ipopt;
using Distributed
using JuMP, Gurobi
using CPUTime


include("C:/MPC_optimization_proxy(2).jl") # trains a NN based optimization proxy, change destination to the folder with MPC_optimization_proxy(2).jl

E1=E1*Eo1[1]
Peng1=Peng1*Pmax
z1=z1*swi
norml=1

E1=E1/norml
Eo1=Eo1/norml
Ptdes1=Ptdes1/norml
Peng1=Peng1/norml

I=size(z1,2)
J=3*I+1
bb=1/1000
bound=10


Pmax=1/norml
Emax=100/norml ## change as per problem
alp=1*norml
bet=10*norml
gam=1.5
tou=5


global xmain=zeros(size(Eo1,1),J)

for i=1:size(Eo1,1)
    for j=1:I+1
        xmain[i,j]=E1[i,j]
    end
    for j=I+2:2*I+1
        xmain[i,j]=Peng1[i,j-(I+1)]
    end
    for j=2*I+2:3*I+1
       xmain[i,j]=z1[i,j-(2*I+1)]
    end
end


epsilon=0
#global xnaught=zeros(k,J)
#xnaught=xmain[1:k,:]

xtest=xmain[end-99:end,:]
Ptdestest=Ptdes1[end-99:end,:]
utest=u[end-99:end,:]

#=
global A=zeros(I,J)
global A1=zeros(I,J)
global A2=zeros(I,J)
global A4=zeros(I,J)
global A5=zeros(I+1,J)=#


OVt=zeros(size(xtest,1))
cxt=zeros(size(xtest,1))
Xtest=zeros(size(xtest,1),J)

global ctrain=zeros(size(arrk,1))
global ctest=zeros(size(arrk,1))
global testerr=zeros(size(arrk,1))
global testobjerr=zeros(size(arrk,1))
global testobjerr1=zeros(size(arrk,1))
global testerrBV=zeros(size(arrk,1))
global testerrCV=zeros(size(arrk,1))

arrswi=zeros(swi+1)

arrswi[1]=0
for i=2:swi+1
    arrswi[i]=(i-1)/swi
end

for l=1:(size(arrk,1))
    #Xtest[:,I+2:2I+1]=Pengval[l,:,:]
    Xtest[:,2I+2:3I+1]=round.(zval[l,:,:])
    for k1=1:size(xtest,1)
        Qin= Model(optimizer_with_attributes(Gurobi.Optimizer))
            @variables(Qin, begin
            Ptbatin[1:I]
            Pmax>=Ptengin[1:I]>=0
            Emax>=Ein[1:I+1]>=0
            end)
            @variable(Qin, swi>=zin[1:I]>=0, Int)
            @constraint(Qin, zin.==(Xtest[k1,2I+2:3I+1]))
            #@constraint(Qin, Ptengin.==(Xtest[k1,I+2:2I+1]))
            @constraint(Qin,Ein[1]==Eo1[1])
                for j=1:I
                    @constraint(Qin, Ein[j+1]==Ein[j]-tou*Ptbatin[j])
                    @constraint(Qin, Ptbatin[j]+Ptengin[j]>=Ptdestest[k1,j])
                    @constraint(Qin, Ptengin[j]<=zin[j]*Pmax/swi)
                end
               
            @objective(Qin, Min, sum(cc[i]*Ein[i] for i=1:I+1)+sum(cc[i+I+1]*Ptengin[i] for i=1:I)+sum(cc[i+2I+1]*zin[i] for i=1:I))
            optimize!(Qin)
            primal_status(Qin)
        if (primal_status(Qin)==NO_SOLUTION)
                OVt[k1]=-9999
        else

            for j=1:I+1
                Xtest[k1,j]=value.(Ein[j])
            end
            for j=I+2:2*I+1
                Xtest[k1,j]=value.(Ptengin[j-(I+1)])
            end
            for j=2I+2:3*I+1
                Xtest[k1,j]=value.(zin[j-(2I+1)])
            end
        
            OVt[k1]=objective_value(Qin)
        end
    end
    

    sum2=zeros(size(xtest,1))
    osum2=zeros(size(xtest,1))
    osum3=zeros(size(xtest,1))

    global ctrain
    global ctest
    global testerr
    global testobjerr
    global testobjerr1
    #global xnaught1=zeros(size(xnaught))
    global Xtest1=zeros(size(Xtest))
    global xtest1=zeros(size(xtest))


    for i=1:(size(xtest,1))
        for j=1:I+1    
            Xtest1[i,j]=Xtest[i,j]/Eo1[1]
            xtest1[i,j]=xtest[i,j]/Eo1[1]
        end
        for j=I+2:3I+1
            Xtest1[i,j]=Xtest[i,j]
            xtest1[i,j]=xtest[i,j]
        end
    end



    sumb=zeros(size(xtest,1))
    sumc=zeros(size(xtest,1))
    for i=1:(size(xtest,1))
        sum2[i]=100*(sum(abs.(Xtest1[i,:]-xtest1[i,:])))/(sum(abs.(xtest1[i,:])))
        osum2[i]=100*abs(cc'*Xtest[i,:]-cc'*xtest[i,:])/abs(cc'*xtest[i,:])
        osum3[i]=100*abs(cc'*Xtest[i,:]-cc'*xtest[i,:])/abs(cc'*xtest[i,:]-cc[I+1]*(Emax))
        sumc[i]=100*(sum(abs.(Xtest1[i,1:2I+2]-xtest1[i,1:2I+2])))/(sum(abs.(xtest1[i,1:2I+2])))
        sumb[i]=100*(sum(abs.(Xtest1[i,2I+2:3I+1]-xtest1[i,2I+2:3I+1])))/(sum(abs.(xtest1[i,2I+2:3I+1])))
        
        if(OVt[i]==-9999)
            global ctest[l]+=1
        end
    end

    
    println("Test error : ", sum(sum2)/(size(xtest,1)) )
    println("Objective Test error : ", sum(osum2)/(size(xtest,1)) )
    testerr[l]=sum(sum2)/(size(xtest,1))
    testerrBV[l]=sum(sumb)/(size(xtest,1))
    testerrCV[l]=sum(sumc)/(size(xtest,1))
    testobjerr[l]=sum(osum2)/(size(xtest,1))
    testobjerr1[l]=sum(osum3)/(size(xtest,1))

end
