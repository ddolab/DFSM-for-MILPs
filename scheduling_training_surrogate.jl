using LinearAlgebra, Random, Gurobi, GAMS, DataFrames, CSV, Printf, JuMP, Ipopt;
using Distributed
using JuMP, Gurobi
using CPUTime


include("data_file1.jl")                                ### change me ###


I=size(Xijt,2) # number of batches
J=size(Xijt,3) # number of units
eta=maximum(Tjt) # Horizon length
T=size(Xijt,4) # number of time slots
V=2J*T+J 
V1=5 # number of added inequalities                     ### change me ###
K=30 # number of training data points                   ### change me ###

bound=10 #the surrogate parameters are bounded between [-bound,bound]
Tj=eta*ones(J)
testset=100 # size of testing data set


global flag=0 
for i=1:size(time1,1)
    if(time1[i]>=7200) # remove data instances from the batch of data for which the optimal solution could not be found within the specified time
        global flag=flag+1
    end
end


normf=eta*10 # normalizing factor for all the model parameters and continuous decision variables (for better training of surrogate parameters)

u=u[1:end-flag]
Xijt=Xijt[1:end-flag,:,:,:]
Sij=Sij[1:end-flag,:,:]/normf
Tjt=Tjt[1:end-flag,:,:]/normf
touij=touij[1:end-flag,:,:]/normf
gamijP=gamijP[1:end-flag,:,:,:]/normf
#gamjtP=gamjtP[1:end-flag,:,:]
#gamsP=gamsP[1:end-flag,:,:]
rhoi=rhoi[1:end-flag,:]/normf
epsi=epsi[1:end-flag,:]/normf
obj=obj[1:end-flag]
time1=time1[1:end-flag]
Tj=Tj/normf
eta=eta/normf


global xmain=zeros(size(u,1),I*J*T+I*J+J*T) # used for converting data from a matrix form in a vector form
global xnaught=zeros(K,I*J*T+I*J+J*T)
global xtest=zeros(testset,I*J*T+I*J+J*T)
global utest=zeros(testset)


for k=1:size(u,1)
    for i=1:I
        for j=1:J
            for t=1:T
                xmain[k,(i-1)*(J*T)+(j-1)*T+t]=Xijt[k,i,j,t]
            end
            xmain[k,I*J*T+(i-1)*J+j]=Sij[k,i,j]
            for t=1:T
                xmain[k,I*J*T+I*J+(j-1)*T+t]=Tjt[k,j,t]
            end
        end
    end

end

normlobj=zeros(size(u,1))   
for k=1:size(u,1)
    normlobj[k]=sum(sum(sum(gamijP[k,i,j,t]*Xijt[k,i,j,t] for i=1:I) for j=1:J) for t=1:T)
end


xnaught=xmain[1:K,:]

xtest=xmain[end-testset+1:end,:]
utest=u[end-testset+1:end,:]

x=zeros(K,I,V)
xt1=zeros(testset,I,V) # test data set
xt=zeros(testset,I,V)
x1=zeros(K,I,V) # training data set

for k=1:K
    for i=1:I
        for j=1:J
            x[k,i,(j-1)*(T)+1:j*T].=xnaught[k,(i-1)*(J*T)+(j-1)*T+1:(i-1)*(J*T)+(j-1)*T+T]
        end
            x[k,i,J*T+1:J*T+J].=xnaught[k,I*J*T+(i-1)*J+1:I*J*T+(i-1)*J+J]
        for j=1:J
            x[k,i,(j-1)*(T)+(J*T+J)+1:j*T+(J*T+J)].=xnaught[k,I*J*T+I*J+(j-1)*T+1:I*J*T+I*J+(j-1)*T+T]
        end
    end
end

for k=1:testset
    for i=1:I
        for j=1:J
            xt[k,i,(j-1)*(T)+1:j*T].=xmain[size(u,1)-testset+k,(i-1)*(J*T)+(j-1)*T+1:(i-1)*(J*T)+(j-1)*T+T]
        end
            xt[k,i,J*T+1:J*T+J].=xmain[size(u,1)-testset+k,I*J*T+(i-1)*J+1:I*J*T+(i-1)*J+J]
        for j=1:J
            xt[k,i,(j-1)*(T)+(J*T+J)+1:j*T+(J*T+J)].=xmain[size(u,1)-testset+k,I*J*T+I*J+(j-1)*T+1:I*J*T+I*J+(j-1)*T+T]
        end
    end
end

for k=1:testset
    for i=1:I
        xt1[k,i,:].=xt[k,i,:]
        if (i>1)
            xt1[k,i,J*T+J+1:end].=0
        end
    end
end
for k=1:K
    for i=1:I
        x1[k,i,:].=x[k,i,:]
        if (i>1)
            x1[k,i,J*T+J+1:end].=0
        end
    end
end

global penalty=zeros(35)  # array to hold the penalty values
global ff1=1e-1*ones(K,I,V)  # scalar parameters associated with penalized constraints corresponding to stationarity conditions
global ff2=1e-1*ones(K,I,V1+J*T-1+J+2+J*(T-1)+J*T) # scalar parameters associated with penalized constraints corresponding to complemnentary slackness conditions
global ff3=1e-1*ones(K,I,V1+J*T-1+J+2+J*(T-1)+J*T) # scalar parameters associated with penalized constraints corresponding to primal feasibility inequalities
global ff4=1e-1*ones(K,I,J+1) # scalar parameters associated with penalized constraints corresponding to primal feasibility equalities
global iter=1
global oiter=1

global cost=zeros(K,I,V)

for k=1:K
    for i=1:I
        for j=1:J
            global cost[k,i,(j-1)*T+1:j*T]=gamijP[k,i,j,:]
        end
    end

    
end

### constraint matrices based on the MILP formulation of this case study

global A1=zeros(1,V)
global A2=ones(1,I)
global A3=zeros(J*T-1,V)
global A4=zeros(J,V)
global A5=zeros(I,J,V)
global A6=zeros(1,V)
global A7=zeros(I,1,V)
global A8=zeros(J,T-1,V)
global A9=zeros(J,T,V)


for j=1:J*T
    global A1[1,j]=1
end
for t=1:J*T-1
    if (t%T!=0)
        global A3[t,J*T+J+t]=1
        global A3[t,J*T+J+t+1]=-1
    end
end

for j=1:J
    global A4[j,J*T+J+j*T]=1
end

for j=1:J
    global A5[:,j,J*T+j].=1
end

global A6[1,J*T+1:J*T+J].=1

global A7[:,1,J*T+1:J*T+J].=1

tt=time()  # to compute the time taken to train the surrogate model
function initialization2(x1)    # solving the nonconvex problem partially to get a initial feasible solution for the BCD algorithm

    Q1= Model(optimizer_with_attributes(Ipopt.Optimizer))
    set_optimizer_attribute(Q1, "mumps_mem_percent", 1000)
    set_optimizer_attribute(Q1, "max_iter", 1)
    @variables(Q1, begin
     mu[1:K,1:I,1:V1+J*T-1+J+2+J*(T-1)+J*T]>=0
     lam[1:K,1:I,1:J+1]
     -bound<=cprime[1:I,1:3,1:V1,1:V]<=bound
     -bound<=cprime1[1:I,1:3,1:V1]<=bound
     A[1:K,1:I,1:V1,1:V]
     AA[1:K,1:I,1:V1,1:V]
    end)
    x=zeros(K,I,V)

    for k=1:K
        for i=1:I
            for j=1:J
                x[k,i,(j-1)*(T)+1:j*T].=x1[k,(i-1)*(J*T)+(j-1)*T+1:(i-1)*(J*T)+(j-1)*T+T]
            end
                x[k,i,J*T+1:J*T+J].=x1[k,I*J*T+(i-1)*J+1:I*J*T+(i-1)*J+J]
            for j=1:J
                x[k,i,(j-1)*(T)+(J*T+J)+1:j*T+(J*T+J)].=x1[k,I*J*T+I*J+(j-1)*T+1:I*J*T+I*J+(j-1)*T+T]
            end
        end
    end
    
        for k=1:K

            for i=1:I
                for j=1:J
                    global A5[i,j,(j-1)*T+1:j*T].=-epsi[k,i]
                end
                for j=1:J
                    global A7[i,1,(j-1)*T+1:j*T].=touij[k,i,j]
                end
                for j=1:J
                    for t=1:T-1
                        global A8[j,t,T*(j-1)+t+1]=eta
                        global A8[j,t,T*(j-1)+J*T+J+t]=1
        
                    end
                    global A8[j,:,J*T+j].=-1
                end
                for j=1:J
                    for t=1:T
                        global A9[j,t,T*(j-1)+t]=eta
                        global A9[j,t,T*(j-1)+J*T+J+t]=-1
        
                    end
                    global A9[j,:,J*T+j].=1
                end
        
            end

            for i=1:I
                for t=1:V1
                    @constraint(Q1,A[k,i,t,:].==(cprime[i,1,t,:]+cprime[i,2,t,:]*u[k]+cprime[i,3,t,:]*u[k]^2))
                end
            end
            

            for i=1:I

                @constraint(Q1, cost[k,i,:].+(A[k,i,:,:]'*mu[k,i,1:V1]).+A1'*lam[k,i,1].+A3'*mu[k,i,V1+1:V1+J*T-1].+A4'*lam[k,i,2:J+1].+A5[i,:,:]'*mu[k,i,V1+J*T:V1+J*T-1+J].-A6'*mu[k,i,V1+J*T-1+J+1].+A7[i,:,:]'*mu[k,i,V1+J*T-1+J+2].+sum(sum(A8[jj,t,:]*mu[k,i,V1+J*T-1+J+2+(jj-1)*(T-1)+t] for t=1:T-1) for jj=1:J).+sum(sum(A9[jj,t,:]*mu[k,i,V1+J*T-1+J+2+J*(T-1)+(jj-1)*T+t] for t=1:T) for jj=1:J).==0)#+(A1'*lam).==0)
                @constraint(Q1,(A[k,i,:,:]*x[k,i,:]).<=(cprime1[i,1,:]+cprime1[i,2,:]*u[k]+cprime1[i,3,:]*u[k]^2))
                @constraint(Q1,(A1*x[k,i,:]).==1)
                @constraint(Q1,(A3*x[k,i,:]).<=0)
                @constraint(Q1,(A4*x[k,i,:]).==Tj)
                @constraint(Q1,(A5[i,:,:]*x[k,i,:]).<=0)
                @constraint(Q1, A6*x[k,i,:].>=rhoi[k,i])
                @constraint(Q1, A7[i,:,:]*x[k,i,:].<=epsi[k,i])
                for j=1:J
                    @constraint(Q1, A8[j,:,:]*x[k,i,:].<=eta)
                    @constraint(Q1, A9[j,:,:]*x[k,i,:].<=eta-touij[k,i,j])
                end
                
                for t=1:V1
                    @constraint(Q1, (mu[k,i,t]*(sum(x[k,i,j]*(A[k,i,t,j]) for j=1:V)-(cprime1[i,1,t]+cprime1[i,2,t]*u[k]+cprime1[i,3,t]*u[k]^2)))==0)
                end
                for t=1:J*T-1
                    @constraint(Q1, (mu[k,i,V1+t]*(sum(x[k,i,j]*(A3[t,j]) for j=1:V)))==0)
                end
                for t=1:J
                    @constraint(Q1, (mu[k,i,V1+J*T-1+t]*(sum(x[k,i,j]*(A5[i,t,j]) for j=1:V)))==0)
                end
                @constraint(Q1, (mu[k,i,V1+J*T-1+J+1]*(sum(x[k,i,j]*(-A6[1,j]) for j=1:V)+rhoi[k,i]))==0)
                @constraint(Q1, (mu[k,i,V1+J*T-1+J+2]*(sum(x[k,i,j]*(A7[i,1,j]) for j=1:V)-epsi[k,i]))==0)
                for jj=1:J
                    for t=1:T-1
                        @constraint(Q1, (mu[k,i,V1+J*T-1+J+2+(jj-1)*(T-1)+t]*(sum(x[k,i,j]*(A8[jj,t,j]) for j=1:V)-eta))==0)
                    end
                    for t=1:T
                        @constraint(Q1, (mu[k,i,V1+J*T-1+J+2+J*(T-1)+(jj-1)*T+t]*(sum(x[k,i,j]*(A9[jj,t,j]) for j=1:V)-eta+touij[k,i,jj]))==0)
                    end
                end

            end

        end

    @objective(Q1, Min, 0)
    optimize!(Q1)
 

    return value.(cprime), value.(cprime1), value.(mu), value.(lam)
end


function decision_block(lam,mu,cprime, cprime1,x,xprev)  ## solving for the decision variables set while keeping the dual variables set and surrogate parameters set fixed
    Q2= Model(optimizer_with_attributes(Gurobi.Optimizer))
    @variables(Q2, begin
    xbar[1:K,1:I,1:V]>=0
    xbar1[1:K,1:I*J*T+I*J+J*T]>=0
    ck2[1:K,1:I,1:V1+J*T-1+J+2+J*(T-1)+J*T]>=0  # penalty for complemnentary conditions
    ck1[1:K,1:I,1:V]>=0   # penalty for stationarity condition
    ck3[1:K,1:I,1:V1+J*T-1+J+2+J*(T-1)+J*T]  # penalty for primal feasibility inequalities
    ck4[1:K,1:I,1:J+1]>=0   # penalty for primal feasibility equalities
    p[1:K,1:I*J*T+I*J+J*T]>=0
    end)
    
    for k=1:K
        for i=1:I
            for j=1:J
                @constraint(Q2, (xbar1[k,(i-1)*(J*T)+(j-1)*T+1:(i-1)*(J*T)+(j-1)*T+T].==xbar[k,i,(j-1)*(T)+1:j*T]))
            end
                @constraint(Q2, xbar1[k,I*J*T+(i-1)*J+1:I*J*T+(i-1)*J+J].==xbar[k,i,J*T+1:J*T+J])
            for j=1:J
                @constraint(Q2, xbar1[k,I*J*T+I*J+(j-1)*T+1:I*J*T+I*J+(j-1)*T+T].==xbar[k,i,(j-1)*(T)+(J*T+J)+1:j*T+(J*T+J)])
            end
        end
    end
    for k=1:K
        for i=1:I
            for j=1:J
                global A5[i,j,(j-1)*T+1:j*T].=-epsi[k,i]
            end
            for j=1:J
                global A7[i,1,(j-1)*T+1:j*T].=touij[k,i,j]
            end
            for j=1:J
                for t=1:T-1
                    global A8[j,t,T*(j-1)+t+1]=eta
                    global A8[j,t,T*(j-1)+J*T+J+t]=1
    
                end
                global A8[j,:,J*T+j].=-1
            end
            for j=1:J
                for t=1:T
                    global A9[j,t,T*(j-1)+t]=eta
                    global A9[j,t,T*(j-1)+J*T+J+t]=-1
    
                end
                global A9[j,:,J*T+j].=1
            end
    
        end
        for j=1:J*T
            @constraint(Q2, A2*xbar[k,:,j].<=1)   
        end
       
        for i=1:I
            @constraint(Q2, xbar[k,i,1:J*T].<=1) 

            @constraint(Q2, cost[k,i,:].+((cprime[i,1,:,:]+cprime[i,2,:,:]*u[k]+cprime[i,3,:,:]*u[k]^2)'*mu[k,i,1:V1]).+A1'*lam[k,i,1].+A3'*mu[k,i,V1+1:V1+J*T-1].+A4'*lam[k,i,2:J+1].+A5[i,:,:]'*mu[k,i,V1+J*T:V1+J*T-1+J].-A6'*mu[k,i,V1+J*T-1+J+1].+A7[i,:,:]'*mu[k,i,V1+J*T-1+J+2].+sum(sum(A8[jj,t,:]*mu[k,i,V1+J*T-1+J+2+(jj-1)*(T-1)+t] for t=1:T-1) for jj=1:J).+sum(sum(A9[jj,t,:]*mu[k,i,V1+J*T-1+J+2+J*(T-1)+(jj-1)*T+t] for t=1:T) for jj=1:J).<=ck1[k,i,:])
            @constraint(Q2, -(cost[k,i,:].+((cprime[i,1,:,:]+cprime[i,2,:,:]*u[k]+cprime[i,3,:,:]*u[k]^2)'*mu[k,i,1:V1]).+A1'*lam[k,i,1].+A3'*mu[k,i,V1+1:V1+J*T-1].+A4'*lam[k,i,2:J+1].+A5[i,:,:]'*mu[k,i,V1+J*T:V1+J*T-1+J].-A6'*mu[k,i,V1+J*T-1+J+1].+A7[i,:,:]'*mu[k,i,V1+J*T-1+J+2].+sum(sum(A8[jj,t,:]*mu[k,i,V1+J*T-1+J+2+(jj-1)*(T-1)+t] for t=1:T-1) for jj=1:J).+sum(sum(A9[jj,t,:]*mu[k,i,V1+J*T-1+J+2+J*(T-1)+(jj-1)*T+t] for t=1:T) for jj=1:J)).<=ck1[k,i,:])
            
            @constraint(Q2,(((cprime[i,1,:,:]+cprime[i,2,:,:]*u[k]+cprime[i,3,:,:]*u[k]^2)*xbar[k,i,:]).-((cprime1[i,1,:]+cprime1[i,2,:]*u[k]+cprime1[i,3,:]*u[k]^2))).<=ck3[k,i,1:V1])
            @constraint(Q2, ck3[k,i,1:V1].>=0)

            @constraint(Q2,(A1*xbar[k,i,:].-1).<=ck4[k,i,1])
            @constraint(Q2,-(A1*xbar[k,i,:].-1).<=ck4[k,i,1])

            @constraint(Q2,(A3*xbar[k,i,:]).<=ck3[k,i,V1+1:V1+J*T-1])
            @constraint(Q2, ck3[k,i,V1+1:V1+J*T-1].>=0)


            @constraint(Q2,(A4*xbar[k,i,:].-Tj).<=ck4[k,i,2:J+1])
            @constraint(Q2,-(A4*xbar[k,i,:].-Tj).<=ck4[k,i,2:J+1])


            @constraint(Q2,(A5[i,:,:]*xbar[k,i,:]).<=ck3[k,i,V1+J*T:V1+J*T-1+J])
            @constraint(Q2, ck3[k,i,V1+J*T:V1+J*T-1+J].>=0)


            @constraint(Q2, (-A6*xbar[k,i,:].+rhoi[k,i]).<=ck3[k,i,V1+J*T-1+J+1])
            @constraint(Q2, ck3[k,i,V1+J*T-1+J+1].>=0)

            @constraint(Q2, (A7[i,:,:]*xbar[k,i,:].-epsi[k,i]).<=ck3[k,i,V1+J*T-1+J+2])
            @constraint(Q2, ck3[k,i,V1+J*T-1+J+2].>=0)


            for jj=1:J
                for t=1:T-1
                    @constraint(Q2, (A8[jj,t,:]'*xbar[k,i,:]-eta)<=ck3[k,i,V1+J*T-1+J+2+(jj-1)*(T-1)+t])
                    @constraint(Q2, ck3[k,i,V1+J*T-1+J+2+(jj-1)*(T-1)+t]>=0)
                end

                for t=1:T
                    @constraint(Q2, (A9[jj,t,:]'*xbar[k,i,:]-eta+touij[k,i,jj])<=ck3[k,i,V1+J*T-1+J+2+J*(T-1)+(jj-1)*T+t])
                    @constraint(Q2, ck3[k,i,V1+J*T-1+J+2+J*(T-1)+(jj-1)*T+t]>=0)
                end
            end

            for t=1:V1
                @constraint(Q2, (mu[k,i,t]*(sum(xbar[k,i,j]*((cprime[i,1,t,j]+cprime[i,2,t,j]*u[k]+cprime[i,3,t,j]*u[k]^2)) for j=1:V)-(cprime1[i,1,t]+cprime1[i,2,t]*u[k]+cprime1[i,3,t]*u[k]^2)))<=ck2[k,i,t])
                @constraint(Q2, -(mu[k,i,t]*(sum(xbar[k,i,j]*((cprime[i,1,t,j]+cprime[i,2,t,j]*u[k]+cprime[i,3,t,j]*u[k]^2)) for j=1:V)-(cprime1[i,1,t]+cprime1[i,2,t]*u[k]+cprime1[i,3,t]*u[k]^2)))<=ck2[k,i,t])
            end


            for t=1:J*T-1
                @constraint(Q2, (mu[k,i,V1+t]*(sum(xbar[k,i,j]*(A3[t,j]) for j=1:V)))<=ck2[k,i,V1+t])
                @constraint(Q2, -(mu[k,i,V1+t]*(sum(xbar[k,i,j]*(A3[t,j]) for j=1:V)))<=ck2[k,i,V1+t])
            end
            for t=1:J
                @constraint(Q2, (mu[k,i,V1+J*T-1+t]*(sum(xbar[k,i,j]*(A5[i,t,j]) for j=1:V)))<=ck2[k,i,V1+J*T-1+t])
                @constraint(Q2, -(mu[k,i,V1+J*T-1+t]*(sum(xbar[k,i,j]*(A5[i,t,j]) for j=1:V)))<=ck2[k,i,V1+J*T-1+t])
            end
            @constraint(Q2, (mu[k,i,V1+J*T-1+J+1]*(sum(xbar[k,i,j]*(-A6[1,j]) for j=1:V)+rhoi[k,i]))<=ck2[k,i,V1+J*T-1+J+1])
            @constraint(Q2, -(mu[k,i,V1+J*T-1+J+1]*(sum(xbar[k,i,j]*(-A6[1,j]) for j=1:V)+rhoi[k,i]))<=ck2[k,i,V1+J*T-1+J+1])

            @constraint(Q2, (mu[k,i,V1+J*T-1+J+2]*(sum(xbar[k,i,j]*(A7[i,1,j]) for j=1:V)-epsi[k,i]))<=ck2[k,i,V1+J*T-1+J+2])
            @constraint(Q2, -(mu[k,i,V1+J*T-1+J+2]*(sum(xbar[k,i,j]*(A7[i,1,j]) for j=1:V)-epsi[k,i]))<=ck2[k,i,V1+J*T-1+J+2])

            for jj=1:J
                for t=1:T-1
                    @constraint(Q2, (mu[k,i,V1+J*T-1+J+2+(jj-1)*(T-1)+t]*(sum(xbar[k,i,j]*(A8[jj,t,j]) for j=1:V)-eta))<=ck2[k,i,V1+J*T-1+J+2+(jj-1)*(T-1)+t])
                    @constraint(Q2, -(mu[k,i,V1+J*T-1+J+2+(jj-1)*(T-1)+t]*(sum(xbar[k,i,j]*(A8[jj,t,j]) for j=1:V)-eta))<=ck2[k,i,V1+J*T-1+J+2+(jj-1)*(T-1)+t])
                end
                for t=1:T
                    @constraint(Q2, (mu[k,i,V1+J*T-1+J+2+J*(T-1)+(jj-1)*T+t]*(sum(xbar[k,i,j]*(A9[jj,t,j]) for j=1:V)-eta+touij[k,i,jj]))<=ck2[k,i,V1+J*T-1+J+2+J*(T-1)+(jj-1)*T+t])
                    @constraint(Q2, -(mu[k,i,V1+J*T-1+J+2+J*(T-1)+(jj-1)*T+t]*(sum(xbar[k,i,j]*(A9[jj,t,j]) for j=1:V)-eta+touij[k,i,jj]))<=ck2[k,i,V1+J*T-1+J+2+J*(T-1)+(jj-1)*T+t])
                end
            end

        end

        for j=1:I*J*T+I*J+J*T
            @constraint(Q2,-(xbar1[k,j]-x[k,j])<=p[k,j])
            @constraint(Q2,(xbar1[k,j]-x[k,j])<=p[k,j])
        end
        

    end

    @objective(Q2, Min, sum(p)+sum(ff1.*ck1)+sum(ff2.*ck2)+sum(ff3.*ck3)+sum(ff4.*ck4)+10^0*(sum((xbar1.-xprev).*(xbar1.-xprev))))  # objective contains the l1-norm of decision varibale loss + total penalty  + proximal term
    optimize!(Q2)
    objv=objective_value(Q2)
    
    return value.(p),value.(xbar1),value.(objv)
    
end

function dual_block(xbar1,cprime,cprime1,x,prevlam,prevmu) ## solving for the dual variables set while keeping the decision variables set and surrogate parameters set fixed
    Q3= Model(optimizer_with_attributes(Gurobi.Optimizer))
    @variables(Q3, begin
    lam[1:K,1:I,1:J+1]
    mu[1:K,1:I,1:V1+J*T-1+J+2+J*(T-1)+J*T]>=0
    ck2[1:K,1:I,1:V1+J*T-1+J+2+J*(T-1)+J*T]>=0  # penalty for complemnentary conditions
    ck1[1:K,1:I,1:V]>=0   # penalty for stationarity condition
    ck3[1:K,1:I,1:V1+J*T-1+J+2+J*(T-1)+J*T]  # penalty for primal feasibility inequalities 
    ck4[1:K,1:I,1:J+1]>=0   # penalty for primal feasibility equalities
    p[1:K,1:I*J*T+I*J+J*T]>=0
    end)
    xbar=zeros(K,I,V)

    for k=1:K
        for i=1:I
            for j=1:J
                xbar[k,i,(j-1)*(T)+1:j*T].=xbar1[k,(i-1)*(J*T)+(j-1)*T+1:(i-1)*(J*T)+(j-1)*T+T]
            end
                xbar[k,i,J*T+1:J*T+J].=xbar1[k,I*J*T+(i-1)*J+1:I*J*T+(i-1)*J+J]
            for j=1:J
                xbar[k,i,(j-1)*(T)+(J*T+J)+1:j*T+(J*T+J)].=xbar1[k,I*J*T+I*J+(j-1)*T+1:I*J*T+I*J+(j-1)*T+T]
            end
        end
    end
    for k=1:K
        for i=1:I
            for j=1:J
                global A5[i,j,(j-1)*T+1:j*T].=-epsi[k,i]
            end
            for j=1:J
                global A7[i,1,(j-1)*T+1:j*T].=touij[k,i,j]
            end
            for j=1:J
                for t=1:T-1
                    global A8[j,t,T*(j-1)+t+1]=eta
                    global A8[j,t,T*(j-1)+J*T+J+t]=1
    
                end
                global A8[j,:,J*T+j].=-1
            end
            for j=1:J
                for t=1:T
                    global A9[j,t,T*(j-1)+t]=eta
                    global A9[j,t,T*(j-1)+J*T+J+t]=-1
    
                end
                global A9[j,:,J*T+j].=1
            end
    
        end
        for i=1:I
            @constraint(Q3, cost[k,i,:].+((cprime[i,1,:,:]+cprime[i,2,:,:]*u[k]+cprime[i,3,:,:]*u[k]^2)'*mu[k,i,1:V1]).+A1'*lam[k,i,1].+A3'*mu[k,i,V1+1:V1+J*T-1].+A4'*lam[k,i,2:J+1].+A5[i,:,:]'*mu[k,i,V1+J*T:V1+J*T-1+J].-A6'*mu[k,i,V1+J*T-1+J+1].+A7[i,:,:]'*mu[k,i,V1+J*T-1+J+2].+sum(sum(A8[jj,t,:]*mu[k,i,V1+J*T-1+J+2+(jj-1)*(T-1)+t] for t=1:T-1) for jj=1:J).+sum(sum(A9[jj,t,:]*mu[k,i,V1+J*T-1+J+2+J*(T-1)+(jj-1)*T+t] for t=1:T) for jj=1:J).<=ck1[k,i,:])
            @constraint(Q3, -(cost[k,i,:].+((cprime[i,1,:,:]+cprime[i,2,:,:]*u[k]+cprime[i,3,:,:]*u[k]^2)'*mu[k,i,1:V1]).+A1'*lam[k,i,1].+A3'*mu[k,i,V1+1:V1+J*T-1].+A4'*lam[k,i,2:J+1].+A5[i,:,:]'*mu[k,i,V1+J*T:V1+J*T-1+J].-A6'*mu[k,i,V1+J*T-1+J+1].+A7[i,:,:]'*mu[k,i,V1+J*T-1+J+2].+sum(sum(A8[jj,t,:]*mu[k,i,V1+J*T-1+J+2+(jj-1)*(T-1)+t] for t=1:T-1) for jj=1:J).+sum(sum(A9[jj,t,:]*mu[k,i,V1+J*T-1+J+2+J*(T-1)+(jj-1)*T+t] for t=1:T) for jj=1:J)).<=ck1[k,i,:])
            
            @constraint(Q3,(((cprime[i,1,:,:]+cprime[i,2,:,:]*u[k]+cprime[i,3,:,:]*u[k]^2)*xbar[k,i,:]).-((cprime1[i,1,:]+cprime1[i,2,:]*u[k]+cprime1[i,3,:]*u[k]^2))).<=ck3[k,i,1:V1])
            @constraint(Q3, ck3[k,i,1:V1].>=0)

            @constraint(Q3,(A1*xbar[k,i,:].-1).<=ck4[k,i,1])
            @constraint(Q3,-(A1*xbar[k,i,:].-1).<=ck4[k,i,1])

            @constraint(Q3,(A3*xbar[k,i,:]).<=ck3[k,i,V1+1:V1+J*T-1])
            @constraint(Q3, ck3[k,i,V1+1:V1+J*T-1].>=0)


            @constraint(Q3,(A4*xbar[k,i,:].-Tj).<=ck4[k,i,2:J+1])
            @constraint(Q3,-(A4*xbar[k,i,:].-Tj).<=ck4[k,i,2:J+1])


            @constraint(Q3,(A5[i,:,:]*xbar[k,i,:]).<=ck3[k,i,V1+J*T:V1+J*T-1+J])
            @constraint(Q3, ck3[k,i,V1+J*T:V1+J*T-1+J].>=0)


            @constraint(Q3, (-A6*xbar[k,i,:].+rhoi[k,i]).<=ck3[k,i,V1+J*T-1+J+1])
            @constraint(Q3, ck3[k,i,V1+J*T-1+J+1].>=0)

            @constraint(Q3, (A7[i,:,:]*xbar[k,i,:].-epsi[k,i]).<=ck3[k,i,V1+J*T-1+J+2])
            @constraint(Q3, ck3[k,i,V1+J*T-1+J+2].>=0)


            for jj=1:J
                for t=1:T-1
                    @constraint(Q3, (A8[jj,t,:]'*xbar[k,i,:]-eta)<=ck3[k,i,V1+J*T-1+J+2+(jj-1)*(T-1)+t])
                    @constraint(Q3, ck3[k,i,V1+J*T-1+J+2+(jj-1)*(T-1)+t]>=0)
                end

                for t=1:T
                    @constraint(Q3, (A9[jj,t,:]'*xbar[k,i,:]-eta+touij[k,i,jj])<=ck3[k,i,V1+J*T-1+J+2+J*(T-1)+(jj-1)*T+t])
                    @constraint(Q3, ck3[k,i,V1+J*T-1+J+2+J*(T-1)+(jj-1)*T+t]>=0)
                end
            end

            for t=1:V1
                @constraint(Q3, (mu[k,i,t]*(sum(xbar[k,i,j]*((cprime[i,1,t,j]+cprime[i,2,t,j]*u[k]+cprime[i,3,t,j]*u[k]^2)) for j=1:V)-(cprime1[i,1,t]+cprime1[i,2,t]*u[k]+cprime1[i,3,t]*u[k]^2)))<=ck2[k,i,t])
                @constraint(Q3, -(mu[k,i,t]*(sum(xbar[k,i,j]*((cprime[i,1,t,j]+cprime[i,2,t,j]*u[k]+cprime[i,3,t,j]*u[k]^2)) for j=1:V)-(cprime1[i,1,t]+cprime1[i,2,t]*u[k]+cprime1[i,3,t]*u[k]^2)))<=ck2[k,i,t])
            end


            for t=1:J*T-1
                @constraint(Q3, (mu[k,i,V1+t]*(sum(xbar[k,i,j]*(A3[t,j]) for j=1:V)))<=ck2[k,i,V1+t])
                @constraint(Q3, -(mu[k,i,V1+t]*(sum(xbar[k,i,j]*(A3[t,j]) for j=1:V)))<=ck2[k,i,V1+t])
            end
            for t=1:J
                @constraint(Q3, (mu[k,i,V1+J*T-1+t]*(sum(xbar[k,i,j]*(A5[i,t,j]) for j=1:V)))<=ck2[k,i,V1+J*T-1+t])
                @constraint(Q3, -(mu[k,i,V1+J*T-1+t]*(sum(xbar[k,i,j]*(A5[i,t,j]) for j=1:V)))<=ck2[k,i,V1+J*T-1+t])
            end
            @constraint(Q3, (mu[k,i,V1+J*T-1+J+1]*(sum(xbar[k,i,j]*(-A6[1,j]) for j=1:V)+rhoi[k,i]))<=ck2[k,i,V1+J*T-1+J+1])
            @constraint(Q3, -(mu[k,i,V1+J*T-1+J+1]*(sum(xbar[k,i,j]*(-A6[1,j]) for j=1:V)+rhoi[k,i]))<=ck2[k,i,V1+J*T-1+J+1])

            @constraint(Q3, (mu[k,i,V1+J*T-1+J+2]*(sum(xbar[k,i,j]*(A7[i,1,j]) for j=1:V)-epsi[k,i]))<=ck2[k,i,V1+J*T-1+J+2])
            @constraint(Q3, -(mu[k,i,V1+J*T-1+J+2]*(sum(xbar[k,i,j]*(A7[i,1,j]) for j=1:V)-epsi[k,i]))<=ck2[k,i,V1+J*T-1+J+2])

            for jj=1:J
                for t=1:T-1
                    @constraint(Q3, (mu[k,i,V1+J*T-1+J+2+(jj-1)*(T-1)+t]*(sum(xbar[k,i,j]*(A8[jj,t,j]) for j=1:V)-eta))<=ck2[k,i,V1+J*T-1+J+2+(jj-1)*(T-1)+t])
                    @constraint(Q3, -(mu[k,i,V1+J*T-1+J+2+(jj-1)*(T-1)+t]*(sum(xbar[k,i,j]*(A8[jj,t,j]) for j=1:V)-eta))<=ck2[k,i,V1+J*T-1+J+2+(jj-1)*(T-1)+t])
                end
                for t=1:T
                    @constraint(Q3, (mu[k,i,V1+J*T-1+J+2+J*(T-1)+(jj-1)*T+t]*(sum(xbar[k,i,j]*(A9[jj,t,j]) for j=1:V)-eta+touij[k,i,jj]))<=ck2[k,i,V1+J*T-1+J+2+J*(T-1)+(jj-1)*T+t])
                    @constraint(Q3, -(mu[k,i,V1+J*T-1+J+2+J*(T-1)+(jj-1)*T+t]*(sum(xbar[k,i,j]*(A9[jj,t,j]) for j=1:V)-eta+touij[k,i,jj]))<=ck2[k,i,V1+J*T-1+J+2+J*(T-1)+(jj-1)*T+t])
                end
            end

        end

      
        for j=1:I*J*T+I*J+J*T
            @constraint(Q3,-(xbar1[k,j]-x[k,j])<=p[k,j])
            @constraint(Q3,(xbar1[k,j]-x[k,j])<=p[k,j])
        end

    end

    @objective(Q3, Min, sum(p)+sum(ff1.*ck1)+sum(ff2.*ck2)+sum(ff3.*ck3)+sum(ff4.*ck4)+10^-2*(sum((mu.-prevmu).*(mu.-prevmu)))+10^-2*(sum((lam.-prevlam).*(lam.-prevlam))))#+10^-2*sum(mu))  ### 10^-2 for prelim results
    optimize!(Q3)
    objv=objective_value(Q3)
    
    return value.(p),value.(lam),value.(mu),value.(objv)
    
end

function surr_param_block(xbar1,lam,mu,x,prevcp,prevcp1) ## solving for the surrogate parameters set while keeping the dual variables set and decision variables set fixed
    Q4= Model(optimizer_with_attributes(Gurobi.Optimizer))
    set_optimizer_attribute(Q4, "BarConvTol", 1e-2)

    @variables(Q4, begin
    -bound<=cprime[1:I,1:3,1:V1,1:V]<=bound
    -bound<=cprime1[1:I,1:3,1:V1]<=bound
    ck2[1:K,1:I,1:V1+J*T-1+J+2+J*(T-1)+J*T]>=0  # penalty for complemnentary conditions
    ck1[1:K,1:I,1:V]>=0   # penalty for stationarity condition
    ck3[1:K,1:I,1:V1+J*T-1+J+2+J*(T-1)+J*T]  # penalty for primal feasibility inequalities
    ck4[1:K,1:I,1:J+1]>=0   # penalty for primal feasibility equalities
    p[1:K,1:I*J*T+I*J+J*T]>=0
    spar[1:I,1:3,1:V1,1:V]>=0
    spar1[1:I,1:3,1:V1]>=0
    end)
    set_start_value.(cprime,prevcp)
    xbar=zeros(K,I,V)

    for k=1:K
        for i=1:I
            for j=1:J
                xbar[k,i,(j-1)*(T)+1:j*T].=xbar1[k,(i-1)*(J*T)+(j-1)*T+1:(i-1)*(J*T)+(j-1)*T+T]
            end
                xbar[k,i,J*T+1:J*T+J].=xbar1[k,I*J*T+(i-1)*J+1:I*J*T+(i-1)*J+J]
            for j=1:J
                xbar[k,i,(j-1)*(T)+(J*T+J)+1:j*T+(J*T+J)].=xbar1[k,I*J*T+I*J+(j-1)*T+1:I*J*T+I*J+(j-1)*T+T]
            end
        end
    end

    for k=1:K
        for i=1:I
            for j=1:J
                global A5[i,j,(j-1)*T+1:j*T].=-epsi[k,i]
            end
            for j=1:J
                global A7[i,1,(j-1)*T+1:j*T].=touij[k,i,j]
            end
            for j=1:J
                for t=1:T-1
                    global A8[j,t,T*(j-1)+t+1]=eta
                    global A8[j,t,T*(j-1)+J*T+J+t]=1
    
                end
                global A8[j,:,J*T+j].=-1
            end
            for j=1:J
                for t=1:T
                    global A9[j,t,T*(j-1)+t]=eta
                    global A9[j,t,T*(j-1)+J*T+J+t]=-1
    
                end
                global A9[j,:,J*T+j].=1
            end
    
        end
        #@constraint(Q4, cprime[:,:,:,J*T+1:end].==0)
        for i=1:I
            @constraint(Q4, cost[k,i,:].+((cprime[i,1,:,:]+cprime[i,2,:,:]*u[k]+cprime[i,3,:,:]*u[k]^2)'*mu[k,i,1:V1]).+A1'*lam[k,i,1].+A3'*mu[k,i,V1+1:V1+J*T-1].+A4'*lam[k,i,2:J+1].+A5[i,:,:]'*mu[k,i,V1+J*T:V1+J*T-1+J].-A6'*mu[k,i,V1+J*T-1+J+1].+A7[i,:,:]'*mu[k,i,V1+J*T-1+J+2].+sum(sum(A8[jj,t,:]*mu[k,i,V1+J*T-1+J+2+(jj-1)*(T-1)+t] for t=1:T-1) for jj=1:J).+sum(sum(A9[jj,t,:]*mu[k,i,V1+J*T-1+J+2+J*(T-1)+(jj-1)*T+t] for t=1:T) for jj=1:J).<=ck1[k,i,:])
            @constraint(Q4, -(cost[k,i,:].+((cprime[i,1,:,:]+cprime[i,2,:,:]*u[k]+cprime[i,3,:,:]*u[k]^2)'*mu[k,i,1:V1]).+A1'*lam[k,i,1].+A3'*mu[k,i,V1+1:V1+J*T-1].+A4'*lam[k,i,2:J+1].+A5[i,:,:]'*mu[k,i,V1+J*T:V1+J*T-1+J].-A6'*mu[k,i,V1+J*T-1+J+1].+A7[i,:,:]'*mu[k,i,V1+J*T-1+J+2].+sum(sum(A8[jj,t,:]*mu[k,i,V1+J*T-1+J+2+(jj-1)*(T-1)+t] for t=1:T-1) for jj=1:J).+sum(sum(A9[jj,t,:]*mu[k,i,V1+J*T-1+J+2+J*(T-1)+(jj-1)*T+t] for t=1:T) for jj=1:J)).<=ck1[k,i,:])
            
            @constraint(Q4,(((cprime[i,1,:,:]+cprime[i,2,:,:]*u[k]+cprime[i,3,:,:]*u[k]^2)*xbar[k,i,:]).-((cprime1[i,1,:]+cprime1[i,2,:]*u[k]+cprime1[i,3,:]*u[k]^2))).<=ck3[k,i,1:V1])
            @constraint(Q4, ck3[k,i,1:V1].>=0)

            @constraint(Q4,(A1*xbar[k,i,:].-1).<=ck4[k,i,1])
            @constraint(Q4,-(A1*xbar[k,i,:].-1).<=ck4[k,i,1])

            @constraint(Q4,(A3*xbar[k,i,:]).<=ck3[k,i,V1+1:V1+J*T-1])
            @constraint(Q4, ck3[k,i,V1+1:V1+J*T-1].>=0)


            @constraint(Q4,(A4*xbar[k,i,:].-Tj).<=ck4[k,i,2:J+1])
            @constraint(Q4,-(A4*xbar[k,i,:].-Tj).<=ck4[k,i,2:J+1])


            @constraint(Q4,(A5[i,:,:]*xbar[k,i,:]).<=ck3[k,i,V1+J*T:V1+J*T-1+J])
            @constraint(Q4, ck3[k,i,V1+J*T:V1+J*T-1+J].>=0)


            @constraint(Q4, (-A6*xbar[k,i,:].+rhoi[k,i]).<=ck3[k,i,V1+J*T-1+J+1])
            @constraint(Q4, ck3[k,i,V1+J*T-1+J+1].>=0)

            @constraint(Q4, (A7[i,:,:]*xbar[k,i,:].-epsi[k,i]).<=ck3[k,i,V1+J*T-1+J+2])
            @constraint(Q4, ck3[k,i,V1+J*T-1+J+2].>=0)


            for jj=1:J
                for t=1:T-1
                    @constraint(Q4, (A8[jj,t,:]'*xbar[k,i,:]-eta)<=ck3[k,i,V1+J*T-1+J+2+(jj-1)*(T-1)+t])
                    @constraint(Q4, ck3[k,i,V1+J*T-1+J+2+(jj-1)*(T-1)+t]>=0)
                end

                for t=1:T
                    @constraint(Q4, (A9[jj,t,:]'*xbar[k,i,:]-eta+touij[k,i,jj])<=ck3[k,i,V1+J*T-1+J+2+J*(T-1)+(jj-1)*T+t])
                    @constraint(Q4, ck3[k,i,V1+J*T-1+J+2+J*(T-1)+(jj-1)*T+t]>=0)
                end
            end

            for t=1:V1
                @constraint(Q4, (mu[k,i,t]*(sum(xbar[k,i,j]*((cprime[i,1,t,j]+cprime[i,2,t,j]*u[k]+cprime[i,3,t,j]*u[k]^2)) for j=1:V)-(cprime1[i,1,t]+cprime1[i,2,t]*u[k]+cprime1[i,3,t]*u[k]^2)))<=ck2[k,i,t])
                @constraint(Q4, -(mu[k,i,t]*(sum(xbar[k,i,j]*((cprime[i,1,t,j]+cprime[i,2,t,j]*u[k]+cprime[i,3,t,j]*u[k]^2)) for j=1:V)-(cprime1[i,1,t]+cprime1[i,2,t]*u[k]+cprime1[i,3,t]*u[k]^2)))<=ck2[k,i,t])
            end


            for t=1:J*T-1
                @constraint(Q4, (mu[k,i,V1+t]*(sum(xbar[k,i,j]*(A3[t,j]) for j=1:V)))<=ck2[k,i,V1+t])
                @constraint(Q4, -(mu[k,i,V1+t]*(sum(xbar[k,i,j]*(A3[t,j]) for j=1:V)))<=ck2[k,i,V1+t])
            end
            for t=1:J
                @constraint(Q4, (mu[k,i,V1+J*T-1+t]*(sum(xbar[k,i,j]*(A5[i,t,j]) for j=1:V)))<=ck2[k,i,V1+J*T-1+t])
                @constraint(Q4, -(mu[k,i,V1+J*T-1+t]*(sum(xbar[k,i,j]*(A5[i,t,j]) for j=1:V)))<=ck2[k,i,V1+J*T-1+t])
            end
            @constraint(Q4, (mu[k,i,V1+J*T-1+J+1]*(sum(xbar[k,i,j]*(-A6[1,j]) for j=1:V)+rhoi[k,i]))<=ck2[k,i,V1+J*T-1+J+1])
            @constraint(Q4, -(mu[k,i,V1+J*T-1+J+1]*(sum(xbar[k,i,j]*(-A6[1,j]) for j=1:V)+rhoi[k,i]))<=ck2[k,i,V1+J*T-1+J+1])

            @constraint(Q4, (mu[k,i,V1+J*T-1+J+2]*(sum(xbar[k,i,j]*(A7[i,1,j]) for j=1:V)-epsi[k,i]))<=ck2[k,i,V1+J*T-1+J+2])
            @constraint(Q4, -(mu[k,i,V1+J*T-1+J+2]*(sum(xbar[k,i,j]*(A7[i,1,j]) for j=1:V)-epsi[k,i]))<=ck2[k,i,V1+J*T-1+J+2])

            for jj=1:J
                for t=1:T-1
                    @constraint(Q4, (mu[k,i,V1+J*T-1+J+2+(jj-1)*(T-1)+t]*(sum(xbar[k,i,j]*(A8[jj,t,j]) for j=1:V)-eta))<=ck2[k,i,V1+J*T-1+J+2+(jj-1)*(T-1)+t])
                    @constraint(Q4, -(mu[k,i,V1+J*T-1+J+2+(jj-1)*(T-1)+t]*(sum(xbar[k,i,j]*(A8[jj,t,j]) for j=1:V)-eta))<=ck2[k,i,V1+J*T-1+J+2+(jj-1)*(T-1)+t])
                end
                for t=1:T
                    @constraint(Q4, (mu[k,i,V1+J*T-1+J+2+J*(T-1)+(jj-1)*T+t]*(sum(xbar[k,i,j]*(A9[jj,t,j]) for j=1:V)-eta+touij[k,i,jj]))<=ck2[k,i,V1+J*T-1+J+2+J*(T-1)+(jj-1)*T+t])
                    @constraint(Q4, -(mu[k,i,V1+J*T-1+J+2+J*(T-1)+(jj-1)*T+t]*(sum(xbar[k,i,j]*(A9[jj,t,j]) for j=1:V)-eta+touij[k,i,jj]))<=ck2[k,i,V1+J*T-1+J+2+J*(T-1)+(jj-1)*T+t])
                end
            end

        end

        for j=1:I*J*T+I*J+J*T
            @constraint(Q4,-(xbar1[k,j]-x[k,j])<=p[k,j])
            @constraint(Q4,(xbar1[k,j]-x[k,j])<=p[k,j])
        end

    end
    @constraint(Q4,-(cprime).<=spar)
    @constraint(Q4,(cprime).<=spar)
    @constraint(Q4,-(cprime1).<=spar1)
    @constraint(Q4,(cprime1).<=spar1)

    @objective(Q4, Min, sum(p)+sum(ff1.*ck1)+sum(ff2.*ck2)+sum(ff3.*ck3)+sum(ff4.*ck4)+10^-2*(sum((cprime.-prevcp).*(cprime.-prevcp)))+10^-2*(sum((cprime1.-prevcp1).*(cprime1.-prevcp1)))+10^-2*(sum(spar)+sum(spar1)))#+sum(spar)+sum(spar1))#+10^-3*sum(spar))  ### 10^-2 for prelim results
    optimize!(Q4)
    objv=objective_value(Q4)
    
    return value.(p),value.(cprime),value.(cprime1),value.(objv),value.(ck1),value.(ck2),value.(ck3),value.(ck4)
    
end  

niter=4  # number of inner iterations

   global pp=zeros(50,niter) # to monitor the convergence of the blocks

   global lam_0=ones(K,I,J+1)
   global mu_0=ones(K,I,V1+J*T-1+J+2+J*(T-1)+J*T)
   global cp_0=ones(I,3,V1,V)
   global cp1_0=ones(I,3,V1)


   global pp1=zeros(K,I*J*T+I*J+J*T)
   global ixbar=zeros(niter,K,I*J*T+I*J+J*T)
   global obj1

   global pp2=zeros(K,I*J*T+I*J+J*T)
   global ilam=zeros(niter,K,I,J+1)
   global imu=zeros(niter,K,I,V1+J*T-1+J+2+J*(T-1)+J*T)
   global obj2

   global pp3=zeros(K,I*J*T+I*J+J*T)
   global icprime=zeros(niter,I,3,V1,V)
   global icprime1=zeros(niter,I,3,V1)
   global obj3
 
   global p1=zeros(K,I,V)
   global p2=zeros(K,I,V1+J*T-1+J+2+J*(T-1)+J*T)
   global p3=zeros(K,I,V1+J*T-1+J+2+J*(T-1)+J*T)
   global p4=zeros(K,I,J+1)

    cp_0,cp1_0,mu_0,lam_0=initialization2(xnaught)
    x_0=xnaught
 
    while (true)
        global oiter
        global iter=1
        while (true)
            global iter
            global oiter
            global pp
            global lam_0
            global mu_0
            global cp_0
            global cp1_0


            global pp1
            global ixbar
            global obj1

            global pp2
            global ilam
            global imu
            global obj2

            global pp3
            global icprime
            global icprime1
            global obj3

            global p1,p2,p3,p4

            if (iter==1)
                pp1,ixbar[iter,:,:],obj1=decision_block(lam_0,mu_0,cp_0,cp1_0,xnaught,x_0)
                pp2,ilam[iter,:,:,:],imu[iter,:,:,:],obj2=dual_block(ixbar[iter,:,:],cp_0,cp1_0,xnaught,lam_0,mu_0)
                pp3,icprime[iter,:,:,:,:],icprime1[iter,:,:,:],obj3,p1,p2,p3,p4=surr_param_block(ixbar[iter,:,:],ilam[iter,:,:,:],imu[iter,:,:,:],xnaught,cp_0,cp1_0)
                iter+=1
            else
                pp1,ixbar[iter,:,:,:],obj1=decision_block(ilam[iter-1,:,:,:],imu[iter-1,:,:,:],icprime[iter-1,:,:,:,:],icprime1[iter-1,:,:,:],xnaught,ixbar[iter-1,:,:,:])
                pp2,ilam[iter,:,:,:],imu[iter,:,:,:],obj2=dual_block(ixbar[iter,:,:,:],icprime[iter-1,:,:,:,:],icprime1[iter-1,:,:,:],xnaught,ilam[iter-1,:,:,:],imu[iter-1,:,:,:])
                println("OITER", oiter)
                pp3,icprime[iter,:,:,:,:],icprime1[iter,:,:,:],obj3,p1,p2,p3,p4=surr_param_block(ixbar[iter,:,:,:],ilam[iter,:,:,:],imu[iter,:,:,:],xnaught,icprime[iter-1,:,:,:,:],icprime1[iter-1,:,:,:])
                pp[oiter,iter-1]=(sum(abs.(ixbar[iter,:,:,:].-ixbar[iter-1,:,:,:]))+sum(abs.(ilam[iter,:,:,:].-ilam[iter-1,:,:,:]))+sum(abs.(imu[iter,:,:,:].-imu[iter-1,:,:,:]))+sum(abs.(icprime[iter,:,:,:,:].-icprime[iter-1,:,:,:,:]))+sum(abs.(icprime1[iter,:,:,:].-icprime1[iter-1,:,:,:])))

                    if (pp[oiter,iter-1]<=1 || iter==niter)
                        cp_0=icprime[iter,:,:,:,:]
                        cp1_0=icprime1[iter,:,:,:]
                        lam_0=ilam[iter,:,:,:]
                        mu_0=imu[iter,:,:,:]
                        break
                    else
                        iter+=1
                    end
            end
        end
        penalty[oiter]=sum(p1)+sum(p2)+sum(p3)+sum(p4)

        mul_fact=5  ## constant to scale up the penalty scalar parameters after each outer iteration
        if (penalty[oiter]<=0.01 || oiter==4)  # run the algorithm until penalty is below a threshold or for a set number of outer iterations
          break
        else
            for k=1:K
                for i=1:I
                    for j=1:V
                        if (p1[k,i,j]>0.1)
                        ff1[k,i,j]=ff1[k,i,j]+(p1[k,i,j])*mul_fact
                        else
                        ff1[k,i,j]=ff1[k,i,j]+(abs(p1[k,i,j]))*1
                        end
                    end
                end
            end
            for k=1:K
                for i=1:I
                    for j=1:V1+J*T-1+J+2+J*(T-1)+J*T
                        if (p2[k,i,j]>0.1)
                            ff2[k,i,j]=ff2[k,i,j]+(p2[k,i,j])*mul_fact
                        else
                            ff2[k,i,j]=ff2[k,i,j]+(abs(p2[k,i,j]))*1
                        end
                    end
                end
            end
            

            for k=1:K
                for i=1:I
                    for j=1:V1+J*T-1+J+2+J*(T-1)+J*T
                        if (p3[k,i,j]>0.1)
                            ff3[k,i,j]=ff3[k,i,j]+(p3[k,i,j])*mul_fact
                        else
                            ff3[k,i,j]=ff3[k,i,j]+(abs(p3[k,i,j]))*1
                        end
                    end
                end
            end
            
            for k=1:K
                for i=1:I
                    for j=1:J+1
                        if (p4[k,i,j]>0.1)
                            ff4[k,i,j]=ff4[k,i,j]+(p4[k,i,j])*mul_fact
                        else
                            ff4[k,i,j]=ff4[k,i,j]+(abs(p4[k,i,j]))*1
                        end
                    end
                end
            end
            
        end

        oiter+=1
    end

An=zeros(K,I,V1,V)
dtt=time()-tt
for k=1:K
    for i=1:I
        for t=1:V1
            for j=1:V
                An[k,i,t,j]=cp_0[i,1,t,j]+cp_0[i,2,t,j]*u[k]+cp_0[i,3,t,j]*u[k]^2
            end
        end
    end
end


X=zeros(K,I,V)
X1=zeros(K,I,V)
OV=zeros(K)
cx=zeros(K)

## running the surrogate on training data set

for k=1:K 
    Q9= Model(optimizer_with_attributes(Gurobi.Optimizer))
    @variable(Q9, xhat[1:I,1:V]>=0)  
    @variable(Q9, 1>=Xijt[1:I,1:J,1:T]>=0)
    @variable(Q9, Sij[1:I,1:J] >=0)
    @variable(Q9, Tjt[1:J,1:T]>=0)
    for i=1:I
        for j=1:J
            @constraint(Q9, xhat[i,(j-1)*(T)+1:j*T].==Xijt[i,j,:])
        end
            @constraint(Q9, xhat[i,J*T+1:J*T+J].==Sij[i,:])
        for j=1:J
            @constraint(Q9, xhat[i,(j-1)*(T)+(J*T+J)+1:j*T+(J*T+J)].==Tjt[j,:])
        end
    end

    for i=1:I
        for j=1:J
            global A5[i,j,(j-1)*T+1:j*T].=-epsi[k,i]
        end
        for j=1:J
            global A7[i,1,(j-1)*T+1:j*T].=touij[k,i,j]
        end
        for j=1:J
            for t=1:T-1
                global A8[j,t,T*(j-1)+t+1]=eta
                global A8[j,t,T*(j-1)+J*T+J+t]=1

            end
            global A8[j,:,J*T+j].=-1
        end
        for j=1:J
            for t=1:T
                global A9[j,t,T*(j-1)+t]=eta
                global A9[j,t,T*(j-1)+J*T+J+t]=-1

            end
            global A9[j,:,J*T+j].=1
        end

    end
    for i=1:I
        @constraint(Q9,An[k,i,:,:]*xhat[i,:].<=cp1_0[i,1,:]+cp1_0[i,2,:]*u[k]+cp1_0[i,3,:]*u[k]^2)
        @constraint(Q9,A3*xhat[i,:].<=0)
        @constraint(Q9,A4*xhat[i,:].==Tj)
        @constraint(Q9,A1*xhat[i,:].==1)
        @constraint(Q9,A5[i,:,:]*xhat[i,:].<=0)
        @constraint(Q9, A6*xhat[i,:].>=rhoi[k,i])
        @constraint(Q9, A7[i,:,:]*xhat[i,:].<=epsi[k,i])
        for j=1:J
            @constraint(Q9, A8[j,:,:]*xhat[i,:].<=eta)
            @constraint(Q9, A9[j,:,:]*xhat[i,:].<=eta-touij[k,i,j])
        end

    end
    for j=1:J*T
        @constraint(Q9, A2*xhat[:,j].<=1)
    end

    

    @objective(Q9, Min, sum(sum(cost[k,i,j].*xhat[i,j] for j=1:V) for i=1:I))

    optimize!(Q9)
    primal_status(Q9)
    if (primal_status(Q9)==NO_SOLUTION)
        
        X[k,:,:]=zeros(I,V)
        OV[k]=0
        X1[k,:,:]=zeros(I,V)
    else
        
        X[k,:,:]=value.(xhat)
        OV[k]=objective_value(Q9)
        X1[k,:,:]=X[k,:,:]
        for i=2:I
            X1[k,i,J*T+J+1:end].=0
        end

    end
end

######################################### feasible projection training ###############################

X2=zeros(K,I,V)
X3=zeros(K,I,V)
OV2=zeros(K)

for k=1:K
    #tt2=time()
    Q10= Model(optimizer_with_attributes(Gurobi.Optimizer))
    #set_optimizer_attribute(Q10, "NonConvex", 2)

    @variable(Q10, Xijt[1:I,1:J,1:T], Bin)
    @variable(Q10, Sij[1:I,1:J]>=0)
    @variable(Q10, Tjt[1:J,1:T]>=0)
    @variable(Q10, xpred[1:I,1:V])
    @variable(Q10, pp[1:I,1:V]>=0)
    for i=1:I
        for j=1:J
            @constraint(Q10, xpred[i,(j-1)*(T)+1:j*T].==Xijt[i,j,:])
        end
            @constraint(Q10, xpred[i,J*T+1:J*T+J].==Sij[i,:])
        for j=1:J
            @constraint(Q10, xpred[i,(j-1)*(T)+(J*T+J)+1:j*T+(J*T+J)].==Tjt[j,:])
        end
    end
    for i=1:I
        for j=1:J
            global A5[i,j,(j-1)*T+1:j*T].=-epsi[k,i]
        end
        for j=1:J
            global A7[i,1,(j-1)*T+1:j*T].=touij[k,i,j]
        end
        for j=1:J
            for t=1:T-1
                global A8[j,t,T*(j-1)+t+1]=eta
                global A8[j,t,T*(j-1)+J*T+J+t]=1

            end
            global A8[j,:,J*T+j].=-1
        end
        for j=1:J
            for t=1:T
                global A9[j,t,T*(j-1)+t]=eta
                global A9[j,t,T*(j-1)+J*T+J+t]=-1

            end
            global A9[j,:,J*T+j].=1
        end

    end
    for i=1:I
        @constraint(Q10,A3*xpred[i,:].<=0)
        @constraint(Q10,A4*xpred[i,:].==Tj)
        @constraint(Q10,A1*xpred[i,:].==1)
        @constraint(Q10,A5[i,:,:]*xpred[i,:].<=0)
        @constraint(Q10, A6*xpred[i,:].>=rhoi[k,i])
        @constraint(Q10, A7[i,:,:]*xpred[i,:].<=epsi[k,i])
        for j=1:J
            @constraint(Q10, A8[j,:,:]*xpred[i,:].<=eta)
            @constraint(Q10, A9[j,:,:]*xpred[i,:].<=eta-touij[k,i,j])
        end
    end
    for j=1:J*T
        @constraint(Q10, A2*xpred[:,j].<=1)
    end
    @constraint(Q10, (xpred.-X[k,:,:]).<=pp)
    @constraint(Q10, -(xpred.-X[k,:,:]).<=pp)


    

    @objective(Q10, Min, sum(pp[:,1:J*T]))

    optimize!(Q10)
    primal_status(Q10)
    
    if (primal_status(Q10)==NO_SOLUTION)
        
        X2[k,:,:]=zeros(I,V)
        OV2[k]=0
        X3[k,:,:]=zeros(I,V)
    else
        
        X2[k,:,:]=value.(xpred)
        OV2[k]=sum(sum(cost[k,i,j].*value.(xpred[i,j]) for j=1:V) for i=1:I)
        X3[k,:,:]=X2[k,:,:]
        for i=2:I
            X3[k,i,J*T+J+1:end].=0
        end

    end
end
                        ######################################### 



global cost_test=zeros(testset,I,V)
for k=1:testset
    for i=1:I
        for j=1:J
           global cost_test[k,i,(j-1)*T+1:j*T]=gamijP[size(u,1)-testset+k,i,j,:]
        end
    end
end

Antest=zeros(testset,I,V1,V)
for k=1:testset
    for i=1:I
        for t=1:V1
            for j=1:V
                Antest[k,i,t,j]=cp_0[i,1,t,j]+cp_0[i,2,t,j]*utest[k]+cp_0[i,3,t,j]*utest[k]^2
            end
        end
    end
end

Xtest=zeros(testset,I,V)
OVt=zeros(testset)
cxt=zeros(testset)
EE=zeros(testset,I)

## running the surrogate on test data set

global tim1=zeros(testset)
global tim2=zeros(testset)

for k=1:testset
    tt2=time()
    Q10= Model(optimizer_with_attributes(Gurobi.Optimizer))
    @variable(Q10, 1e1>=ee[1:I]>=-1e1)

    @variable(Q10, 1>=Xijt[1:I,1:J,1:T]>=0)
    @variable(Q10, Sij[1:I,1:J]>=0)
    @variable(Q10, Tjt[1:J,1:T]>=0)
    @variable(Q10, xpred[1:I,1:V])
    for i=1:I
        for j=1:J
            @constraint(Q10, xpred[i,(j-1)*(T)+1:j*T].==Xijt[i,j,:])
        end
            @constraint(Q10, xpred[i,J*T+1:J*T+J].==Sij[i,:])
        for j=1:J
            @constraint(Q10, xpred[i,(j-1)*(T)+(J*T+J)+1:j*T+(J*T+J)].==Tjt[j,:])
        end
    end
    for i=1:I
        for j=1:J
            global A5[i,j,(j-1)*T+1:j*T].=-epsi[size(u,1)-testset+k,i]
        end
        for j=1:J
            global A7[i,1,(j-1)*T+1:j*T].=touij[size(u,1)-testset+k,i,j]
        end
        for j=1:J
            for t=1:T-1
                global A8[j,t,T*(j-1)+t+1]=eta
                global A8[j,t,T*(j-1)+J*T+J+t]=1

            end
            global A8[j,:,J*T+j].=-1
        end
        for j=1:J
            for t=1:T
                global A9[j,t,T*(j-1)+t]=eta
                global A9[j,t,T*(j-1)+J*T+J+t]=-1

            end
            global A9[j,:,J*T+j].=1
        end

    end
    for i=1:I
        @constraint(Q10,Antest[k,i,:,:]*xpred[i,:].<=cp1_0[i,1,:]+cp1_0[i,2,:]*utest[k]+cp1_0[i,3,:]*utest[k]^2+ee[i]) #<=b3[i,:]
        @constraint(Q10,A3*xpred[i,:].<=0)
        @constraint(Q10,A4*xpred[i,:].==Tj)
        @constraint(Q10,A1*xpred[i,:].==1)
        @constraint(Q10,A5[i,:,:]*xpred[i,:].<=0)
        @constraint(Q10, A6*xpred[i,:].>=rhoi[size(u,1)-testset+k,i])
        @constraint(Q10, A7[i,:,:]*xpred[i,:].<=epsi[size(u,1)-testset+k,i])
        for j=1:J
            @constraint(Q10, A8[j,:,:]*xpred[i,:].<=eta)
            @constraint(Q10, A9[j,:,:]*xpred[i,:].<=eta-touij[size(u,1)-testset+k,i,j])
        end

    end
    for j=1:J*T
            @constraint(Q10, A2*xpred[:,j].<=1)
    end

    

    @objective(Q10, Min, sum(sum(sum(gamijP[size(u,1)-testset+k,i,j,t]*Xijt[i,j,t] for i=1:I) for j=1:J) for t=1:T)+sum(ee.*ee))#+10^5*sum(xpred[:,1:J*T]))#+10^2*sum(xpred[:,1:J*T]))

    optimize!(Q10)
    primal_status(Q10)
    global tim2[k]=time()-tt2
    
    if (primal_status(Q10)==NO_SOLUTION)
        
        Xtest[k,:,:]=zeros(I,V)
        OVt[k]=0
    else
        
        Xtest[k,:,:]=value.(xpred)
        OVt[k]=objective_value(Q10)-sum(value.(ee).*value.(ee))
        EE[k,:].=value.(ee)

    end
end

######################################### feasible projection ###############################

Xtest2=zeros(testset,I,V)
Xtest3=zeros(testset,I,V)
OVt2=zeros(testset)

for k=1:testset
    println("current instance", k)
    tt2=time()
    Q10= Model(optimizer_with_attributes(Gurobi.Optimizer))
    #set_optimizer_attribute(Q10, "NonConvex", 2)
    set_optimizer_attribute(Q10, "MIPGap", 0.01)
    set_optimizer_attribute(Q10, "TIME_LIMIT", 30)

    @variable(Q10, 1>=Xijt[1:I,1:J,1:T]>=0, Bin)
    @variable(Q10, Sij[1:I,1:J] >=0)
    @variable(Q10, Tjt[1:J,1:T]>=0)
    @variable(Q10, xpred[1:I,1:V])
    @variable(Q10, pp[1:I,1:V]>=0)
    for i=1:I
        for j=1:J
            @constraint(Q10, xpred[i,(j-1)*(T)+1:j*T].==Xijt[i,j,:])
        end
            @constraint(Q10, xpred[i,J*T+1:J*T+J].==Sij[i,:])
        for j=1:J
            @constraint(Q10, xpred[i,(j-1)*(T)+(J*T+J)+1:j*T+(J*T+J)].==Tjt[j,:])
        end
    end
    for i=1:I
        for j=1:J
            global A5[i,j,(j-1)*T+1:j*T].=-epsi[size(u,1)-testset+k,i]
        end
        for j=1:J
            global A7[i,1,(j-1)*T+1:j*T].=touij[size(u,1)-testset+k,i,j]
        end
        for j=1:J
            for t=1:T-1
                global A8[j,t,T*(j-1)+t+1]=eta
                global A8[j,t,T*(j-1)+J*T+J+t]=1

            end
            global A8[j,:,J*T+j].=-1
        end
        for j=1:J
            for t=1:T
                global A9[j,t,T*(j-1)+t]=eta
                global A9[j,t,T*(j-1)+J*T+J+t]=-1

            end
            global A9[j,:,J*T+j].=1
        end

    end
    for i=1:I
        @constraint(Q10,A3*xpred[i,:].<=0)
        @constraint(Q10,A4*xpred[i,:].==Tj)
        @constraint(Q10,A1*xpred[i,:].==1)
        @constraint(Q10,A5[i,:,:]*xpred[i,:].<=0)
        @constraint(Q10, A6*xpred[i,:].>=rhoi[size(u,1)-testset+k,i])
        @constraint(Q10, A7[i,:,:]*xpred[i,:].<=epsi[size(u,1)-testset+k,i])
        for j=1:J
            @constraint(Q10, A8[j,:,:]*xpred[i,:].<=eta)
            @constraint(Q10, A9[j,:,:]*xpred[i,:].<=eta-touij[size(u,1)-testset+k,i,j])
        end
    end
    for j=1:J*T
        @constraint(Q10, A2*xpred[:,j].<=1)
    end
    @constraint(Q10, (xpred.-Xtest[k,:,:]).<=pp)
    @constraint(Q10, -(xpred.-Xtest[k,:,:]).<=pp)


    

    @objective(Q10, Min, sum(pp[:,1:J*T]))

    optimize!(Q10)
    primal_status(Q10)
    global tim1[k]=time()-tt2
    
    if (primal_status(Q10)==NO_SOLUTION)
        
        Xtest2[k,:,:]=zeros(I,V)
        OVt2[k]=0
        Xtest3[k,:,:]=zeros(I,V)
    else
        
        Xtest2[k,:,:]=value.(xpred)
        OVt2[k]=sum(sum(cost_test[k,i,j].*value.(xpred[i,j]) for j=1:V) for i=1:I)
        Xtest3[k,:,:]=Xtest2[k,:,:]
        for i=2:I
            Xtest3[k,i,J*T+J+1:end].=0
        end

    end
end
                ######################################### 

## computing the linear relaxation solution for the test data set and using projection to find a integer feasible solution

    Xtest1=zeros(testset,I,V)
    Xtest4=zeros(testset,I,V)
    OVt1=zeros(testset)
    for k=1:testset
        Q10= Model(optimizer_with_attributes(Gurobi.Optimizer))
        @variable(Q10, 1>=Xijt[1:I,1:J,1:T]>=0)
        @variable(Q10, Sij[1:I,1:J] >=0)
        @variable(Q10, Tjt[1:J,1:T]>=0)
        @variable(Q10, xpred[1:I,1:V])
        for i=1:I
            for j=1:J
                @constraint(Q10, xpred[i,(j-1)*(T)+1:j*T].==Xijt[i,j,:])
            end
                @constraint(Q10, xpred[i,J*T+1:J*T+J].==Sij[i,:])
            for j=1:J
                @constraint(Q10, xpred[i,(j-1)*(T)+(J*T+J)+1:j*T+(J*T+J)].==Tjt[j,:])
            end
        end

        for i=1:I
            for j=1:J
                global A5[i,j,(j-1)*T+1:j*T].=-epsi[size(u,1)-testset+k,i]
            end
            for j=1:J
                global A7[i,1,(j-1)*T+1:j*T].=touij[size(u,1)-testset+k,i,j]
            end
            for j=1:J
                for t=1:T-1
                    global A8[j,t,T*(j-1)+t+1]=eta
                    global A8[j,t,T*(j-1)+J*T+J+t]=1

                end
                global A8[j,:,J*T+j].=-1
            end
            for j=1:J
                for t=1:T
                    global A9[j,t,T*(j-1)+t]=eta
                    global A9[j,t,T*(j-1)+J*T+J+t]=-1

                end
                global A9[j,:,J*T+j].=1
            end

        end
        for i=1:I
            @constraint(Q10,A3*xpred[i,:].<=0)
            @constraint(Q10,A4*xpred[i,:].==Tj)
            @constraint(Q10,A1*xpred[i,:].==1)
            @constraint(Q10,A5[i,:,:]*xpred[i,:].<=0)
            @constraint(Q10, A6*xpred[i,:].>=rhoi[size(u,1)-testset+k,i])
            @constraint(Q10, A7[i,:,:]*xpred[i,:].<=epsi[size(u,1)-testset+k,i])
            for j=1:J
                @constraint(Q10, A8[j,:,:]*xpred[i,:].<=eta)
                @constraint(Q10, A9[j,:,:]*xpred[i,:].<=eta-touij[size(u,1)-testset+k,i,j])
            end

        end
        for j=1:J*T
            @constraint(Q10, A2*xpred[:,j].<=1)
        end

        @objective(Q10, Min, sum(sum(sum(gamijP[size(u,1)-testset+k,i,j,t]*Xijt[i,j,t] for i=1:I) for j=1:J) for t=1:T))

        optimize!(Q10)
        primal_status(Q10)
        
        if (primal_status(Q10)==NO_SOLUTION)
            
            Xtest1[k,:,:]=zeros(I,V)
            OVt1[k]=0
            Xtest4[k,:,:]=zeros(I,V)
        else
            
            Xtest1[k,:,:]=value.(xpred)
            OVt1[k]=sum(sum(cost_test[k,i,j].*value.(xpred[i,j]) for j=1:V) for i=1:I)
            Xtest4[k,:,:]=Xtest1[k,:,:]
            for i=2:I
                Xtest4[k,i,J*T+J+1:end].=0
            end
        end
    end



######################################### feasible projection ###############################   


for k=1:testset
    tt2=time()
    Q10= Model(optimizer_with_attributes(Gurobi.Optimizer))
    set_optimizer_attribute(Q10, "MIPGap", 0.01)
    set_optimizer_attribute(Q10, "TIME_LIMIT", 30)

    @variable(Q10, 1>=Xijt[1:I,1:J,1:T]>=0, Bin)
    @variable(Q10, Sij[1:I,1:J] >=0)
    @variable(Q10, Tjt[1:J,1:T]>=0)
    @variable(Q10, xpred[1:I,1:V])
    @variable(Q10, pp[1:I,1:V]>=0)
    for i=1:I
        for j=1:J
            @constraint(Q10, xpred[i,(j-1)*(T)+1:j*T].==Xijt[i,j,:])
        end
            @constraint(Q10, xpred[i,J*T+1:J*T+J].==Sij[i,:])
        for j=1:J
            @constraint(Q10, xpred[i,(j-1)*(T)+(J*T+J)+1:j*T+(J*T+J)].==Tjt[j,:])
        end
    end
    for i=1:I
        for j=1:J
            global A5[i,j,(j-1)*T+1:j*T].=-epsi[size(u,1)-testset+k,i]
        end
        for j=1:J
            global A7[i,1,(j-1)*T+1:j*T].=touij[size(u,1)-testset+k,i,j]
        end
        for j=1:J
            for t=1:T-1
                global A8[j,t,T*(j-1)+t+1]=eta
                global A8[j,t,T*(j-1)+J*T+J+t]=1

            end
            global A8[j,:,J*T+j].=-1
        end
        for j=1:J
            for t=1:T
                global A9[j,t,T*(j-1)+t]=eta
                global A9[j,t,T*(j-1)+J*T+J+t]=-1

            end
            global A9[j,:,J*T+j].=1
        end

    end
    for i=1:I
        @constraint(Q10,A3*xpred[i,:].<=0)
        @constraint(Q10,A4*xpred[i,:].==Tj)
        @constraint(Q10,A1*xpred[i,:].==1)
        @constraint(Q10,A5[i,:,:]*xpred[i,:].<=0)
        @constraint(Q10, A6*xpred[i,:].>=rhoi[size(u,1)-testset+k,i])
        @constraint(Q10, A7[i,:,:]*xpred[i,:].<=epsi[size(u,1)-testset+k,i])
        for j=1:J
            @constraint(Q10, A8[j,:,:]*xpred[i,:].<=eta)
            @constraint(Q10, A9[j,:,:]*xpred[i,:].<=eta-touij[size(u,1)-testset+k,i,j])
        end
    end
    for j=1:J*T
        @constraint(Q10, A2*xpred[:,j].<=1)
    end
    @constraint(Q10, (xpred.-Xtest1[k,:,:]).<=pp)
    @constraint(Q10, -(xpred.-Xtest1[k,:,:]).<=pp)

    

    @objective(Q10, Min, sum(pp[:,1:J*T]))

    optimize!(Q10)
    primal_status(Q10)

    
    if (primal_status(Q10)==NO_SOLUTION)
        
        Xtest1[k,:,:]=zeros(I,V)
        OVt1[k]=0
        Xtest4[k,:,:]=zeros(I,V)
    else
        
        Xtest1[k,:,:]=value.(xpred)
        OVt1[k]=sum(sum(cost_test[k,i,j].*value.(xpred[i,j]) for j=1:V) for i=1:I)
        Xtest4[k,:,:]=Xtest1[k,:,:]
        for i=2:I
            Xtest4[k,i,J*T+J+1:end].=0
        end

    end
end


######################################### 


trainerr_bf=zeros(K) # training decision error before feasibility restoration
trainoberr_bf=zeros(K) # training objective error before feasibility restoration
trainerr_pf=zeros(K) # training error post feasibility restoration
trainoberr_pf=zeros(K) # training objective error post feasibility restoration
testerr_pf=zeros(testset) # test decision error post feasibility restoration
testoberr_pf=zeros(testset) # test objective error post feasibility restoration
LRerr_pf=zeros(testset) # Linear relaxation decision error post feasibility restoration
LRoberr_pf=zeros(testset) # Linear relaxation objective error post feasibility restoration



global ctrain=0 # to keep a count if any infeasible instances resulted from using the surrogate model
global ctrain1=0
global ctest=0
global clin=0


for i=1:K
    trainerr_bf[i]=100*(sum(abs.(X1[i,:,J*T+1:2*J*T+J].-x1[i,:,J*T+1:2*J*T+J])))/(sum(abs.(x1[i,:,J*T+1:2*J*T+J])))
    trainoberr_bf[i]=100*(abs.(OV[i]-normlobj[i]))/normlobj[i]
    if(OV[i]==0)
        trainerr_bf[i]=0
        trainoberr_bf[i]=0
        global ctrain+=1
    end
end


for i=1:K
    trainerr_pf[i]=100*(sum(abs.(X3[i,:,J*T+1:2*J*T+J].-x1[i,:,J*T+1:2*J*T+J])))/(sum(abs.(x1[i,:,J*T+1:2*J*T+J])))
    trainoberr_pf[i]=100*(abs.(OV2[i]-normlobj[i]))/normlobj[i]
    if(OV2[i]==0)
        trainerr_pf[i]=0
        trainoberr_pf[i]=0
        global ctrain1+=1
    end
end


for i=1:testset
    testerr_pf[i]=100*(sum(abs.(Xtest3[i,:,J*T+1:2*J*T+J].-xt1[i,:,J*T+1:2*J*T+J])))/(sum(abs.(xt1[i,:,J*T+1:2*J*T+J])))
    testoberr_pf[i]=100*(abs.(OVt2[i]-normlobj[end-testset+i]))/normlobj[end-testset+i]
    if(OVt2[i]==0)
        testerr_pf[i]=0
        testoberr_pf[i]=0
        global ctest+=1
    end
end

for i=1:testset
    LRerr_pf[i]=100*(sum(abs.(Xtest4[i,:,J*T+1:2*J*T+J]-xt1[i,:,J*T+1:2*J*T+J])))/(sum(abs.(xt1[i,:,J*T+1:2*J*T+J])))
    LRoberr_pf[i]=100*(abs.(OVt1[i]-normlobj[end-testset+i]))/normlobj[end-testset+i]
     if(OVt1[i]==0)
        LRerr_pf[i]=0
        LRoberr_pf[i]=0
        global clin+=1
    end
end

XX1=zeros(K,I,J)
YY1=zeros(K,I,J)
for k=1:K
   for i=1:I
       for j=1:J
            XX1[k,i,j]=sum(X1[k,i,(j-1)*T+1:j*T])
       end
   end
end

for k=1:K
   for i=1:I
       for j=1:J
            YY1[k,i,j]=sum(x1[k,i,(j-1)*T+1:j*T])
       end
   end
end


XX2=zeros(K,I,J)
YY2=zeros(K,I,J)
for k=1:K
   for i=1:I
       for j=1:J
            XX2[k,i,j]=sum(X3[k,i,(j-1)*T+1:j*T])
       end
   end
end

for k=1:K
   for i=1:I
       for j=1:J
            YY2[k,i,j]=sum(x1[k,i,(j-1)*T+1:j*T])
       end
   end
end


XX=zeros(testset,I,J)
YY=zeros(testset,I,J)
for k=1:testset
   for i=1:I
       for j=1:J
            XX[k,i,j]=sum(Xtest3[k,i,(j-1)*T+1:j*T])
       end
   end
end

for k=1:testset
   for i=1:I
       for j=1:J
            YY[k,i,j]=sum(xt1[k,i,(j-1)*T+1:j*T])
       end
   end
end



if (K==3)
    str="w"
else
    str="a"
end
output_file=open("results/result_1.jl",str)  ## change destination ##
write(output_file, "|K|= ") 
show(output_file,  K) 
write(output_file, "; \n \n")

write(output_file, "time= ") 
show(output_file,  dtt) 
write(output_file, "; \n \n")

write(output_file, "Training erro bf = ") 
show(output_file, sum(s1)/(K-ctrain)) 
write(output_file, "; \n \n")


write(output_file, "Objective train error bf = ") 
show(output_file,  sum(o1)/(K-ctrain)) 
write(output_file, "; \n \n")


write(output_file, "BVR train bf = ") 
show(output_file, sum(abs.(XX1.-YY1))*100/sum(YY1)) 
write(output_file, "; \n \n")



write(output_file, "Training error pf = ") 
show(output_file, sum(s2)/(K-ctrain1)) 
write(output_file, "; \n \n")

write(output_file, "Objective training error pf = ") 
show(output_file, sum(o2)/(K-ctrain1)) 
write(output_file, "; \n \n")


write(output_file, "BVR train pf = ") 
show(output_file, sum(abs.(XX2.-YY2))*100/sum(YY2)) 
write(output_file, "; \n \n")



write(output_file, "Test error= ") 
show(output_file, sum(sum2)/(testset-ctest)) 
write(output_file, "; \n \n")


write(output_file, "Objective Test error = ") 
show(output_file, sum(osum2)/(testset-ctest)) 
write(output_file, "; \n \n")


write(output_file, "Binary variables error = ") 
show(output_file, sum(abs.(XX.-YY))*100/sum(YY)) 
write(output_file, "; \n \n")


write(output_file, "Continuous variables error = ") 
show(output_file, sum(abs.(Xtest3[:,:,J*T+1:2*J*T+J].-xt1[:,:,J*T+1:2*J*T+J]))*100/sum(abs.(xt1[:,:,J*T+1:2*J*T+J]))) 
write(output_file, "; \n \n")


write(output_file, "Penalty ") 
show(output_file, penalty) 
write(output_file, "; \n \n")


write(output_file, "ctest= ") 
show(output_file, ctest) 
write(output_file, "; \n \n \n \n")

write(output_file, "totim= ") 
show(output_file, tot_tim) 
write(output_file, "; \n \n \n \n")

write(output_file, "OVt= ") 
show(output_file, OVt2) 
write(output_file, "; \n \n \n \n")
close(output_file)


output_file=open("outputs/outputs_1.jl",str) ## change destination ##
write(output_file, "sum2= ") 
show(output_file,  sum2) 
write(output_file, "; \n \n")

write(output_file, "osum2= ") 
show(output_file,  osum2) 
write(output_file, "; \n \n")

write(output_file, "Xtest= ") 
show(output_file,  Xtest) 
write(output_file, "; \n \n")

write(output_file, "Xtest3= ") 
show(output_file, Xtest3) 
write(output_file, "; \n \n")

write(output_file, "tim1= ") 
show(output_file, tim1) 
write(output_file, "; \n \n")
close(output_file)
