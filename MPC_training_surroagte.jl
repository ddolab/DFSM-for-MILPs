using LinearAlgebra, Random, Gurobi, GAMS, DataFrames, CSV, Printf, BARON, JuMP, Ipopt;
using Distributed
using JuMP, Gurobi
using CPUTime


include(".jl")                                ### change me ###

I=30 # Horizon length
J=3*I+1 # number of decision variables
k=10 # training data set size                                           ### change me ###
V=30 # number of additional inequalities in the surrogate model (2T)    ### change me ###
bb=1/1000 
bound=10
V1=V

## specified problem parameters
Pmax=1
Emax=100
alp=1
bet=10
gam=1.5
tou=5

init=time()

global xmain=zeros(size(Eo1,1),J)
global penalty=zeros(35) # array to hold the penalty values
global ff1=1e-1*ones(k,J)   # scalar parameters associated with penalized constraints corresponding to stationarity conditions
global ff2=1e-1*ones(k,I+V+1) # scalar parameters associated with penalized constraints corresponding to complemnentary slackness conditions
global ff3=1e-1*ones(k,I+V+1) # scalar parameters associated with penalized constraints corresponding to primal feasibility inequalities
global ff4=1e-1*ones(k,I) # scalar parameters associated with penalized constraints corresponding to primal feasibility equalities
global iter=1
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


global xnaught=zeros(k,J)
xnaught=xmain[1:k,:]

xtest=xmain[end-99:end,:]
Ptdestest=Ptdes1[end-99:end,:]
utest=u[end-99:end,:]

global A=zeros(I,J)
global A1=zeros(I,J)
global A2=zeros(I,J)
global A4=zeros(I,J)
global A5=zeros(I+1,J)

epsilon=0 # 0 when want to solve an LP surroagte model (0, 0.1] when want to solve for a QP surrogate model
global oiter=1

## constraint matrices based on the MILP formulation of this case study
for i=1:I
    for j=1:J
        global A[i,i]=-1/tou
        global A[i,i+1]=1/tou     ## -Et/tou + Et+1/tou -Pteng <= -Ptdes
        global A[i,I+1+i]=-1  

        global A1[i,1*(I)+1+i]=1   # Peng<=z Pmax
        global A1[i,2*(I)+1+i]=-Pmax/swi

        global A4[i,1]=1
        if(i==j)
            global A5[i,j]=1
        end
    end
end
global A5[I+1,I+1]=1

OVin=zeros(size(xtest,1))
global xin=zeros(size(xtest,1),J)
for i=1:size(xtest,1)
    Qin= Model(optimizer_with_attributes(Gurobi.Optimizer))
        @variables(Qin, begin
        Ptbatin[1:I]
        Pmax>=Ptengin[1:I]>=0
        Emax>=Ein[1:I+1]>=0
        end)
        @variable(Qin, swi>=zin[1:I]>=0)
        @constraint(Qin,Ein[1]==Eo1[1])
            for j=1:I
                @constraint(Qin, Ein[j+1]==Ein[j]-tou*Ptbatin[j])
                @constraint(Qin, Ptbatin[j]+Ptengin[j]>=Ptdestest[i,j])
                @constraint(Qin, Ptengin[j]<=zin[j]*Pmax/swi)
            end

        @objective(Qin, Min, sum(cc[i]*Ein[i] for i=1:I+1)+sum(cc[i+I+1]*Ptengin[i] for i=1:I)+sum(cc[i+2I+1]*zin[i] for i=1:I))
        
        optimize!(Qin)

        for j=1:I+1
            xin[i,j]=value.(Ein[j])
        end
        for j=I+2:2*I+1
            xin[i,j]=value.(Ptengin[j-(I+1)])
        end
        for j=2*I+2:3*I+1
           xin[i,j]=value.(zin[j-(2*I+1)])
        end
        OVin[i]=objective_value(Qin)
end

xin[:,2I+2:3I+1]=round.(xin[:,2I+2:3I+1])

for i=1:size(xtest,1)
    Qin= Model(optimizer_with_attributes(Gurobi.Optimizer))
        @variables(Qin, begin
        Ptbatin[1:I]
        Pmax>=Ptengin[1:I]>=0
        Emax>=Ein[1:I+1]>=0
        end)
        @variable(Qin, swi>=zin[1:I]>=0)
        @constraint(Qin, zin.==xin[i,2I+2:3I+1])
        @constraint(Qin,Ein[1]==Eo1[1])
            for j=1:I
                @constraint(Qin, Ein[j+1]==Ein[j]-tou*Ptbatin[j])
                @constraint(Qin, Ptbatin[j]+Ptengin[j]>=Ptdestest[i,j])
                @constraint(Qin, Ptengin[j]<=zin[j]*Pmax/swi)
            end

        @objective(Qin, Min, sum(cc[i]*Ein[i] for i=1:I+1)+sum(cc[i+I+1]*Ptengin[i] for i=1:I)+sum(cc[i+2I+1]*zin[i] for i=1:I))
        
        optimize!(Qin)

        for j=1:I+1
            xin[i,j]=value.(Ein[j])
        end
        for j=I+2:2*I+1
            xin[i,j]=value.(Ptengin[j-(I+1)])
        end
        for j=2*I+2:3*I+1
           xin[i,j]=value.(zin[j-(2*I+1)])
        end
        OVin[i]=objective_value(Qin)
end

#tt=time()

    function initialization2(x)    # solving the nonconvex problem partially to get a initial feasible solution for the BCD algorithm
        V1=1  
        counter=1
        Q1= Model(optimizer_with_attributes(Ipopt.Optimizer))
        set_optimizer_attribute(Q1, "mumps_mem_percent", 100)
        set_optimizer_attribute(Q1, "max_iter", 1)
        @variables(Q1, begin
         lam[1:k,1:I]
         mu[1:k,1:I+1+V]>=0
         -bound<=cprime[1:4,1:V1,1:J]<=bound
         c[1:k,1:J]
         A[1:k,1:V,1:J]
        end)
        
    
            for i=1:k
                for j=1:3I+1
                    @constraint(Q1, c[i,j]==cc[j])  
                end
                
                    @constraint(Q1,(c[i,:].+epsilon*x[i,:]).+A5'*mu[i,1:I+1].+A4'*lam[i,:].==0)
    
                    @constraint(Q1,A4*x[i,:].==Eo1[i])
                    @constraint(Q1,A5*x[i,:].<=Emax)
    
                    for j=1:I+1
                        @constraint(Q1,mu[i,j]*(A5[j,:]'*x[i,:].-Emax).==0)  
                    end
        
            end
    
        @objective(Q1, Min, 0)
        optimize!(Q1)
        
        return value.(cprime[:,1,:]),value.(lam), value.(mu)
    end

    function decision_block(lam,mu,cprime,x,xprev) ## solving for the decision variables set while keeping the dual variables set and surrogate parameters set fixed
        Q2= Model(optimizer_with_attributes(Gurobi.Optimizer))
        @variables(Q2, begin
        xbar[1:k,1:J]>=0
        ck2[1:k,1:I+1+V]>=0  # penalty for complemnentary conditions
        ck1[1:k,1:J]>=0   # penalty for stationarity condition
        ck3[1:k,1:I+1+V]  # penalty for primal feasibility inequalities 
        ck4[1:k,1:I]>=0   # penalty for primal feasibility equalities
        p[1:k,1:3I+1]>=0
        c[1:k,1:J]
        end)
        for i=1:k
            for j=I+2:2I+1    
               @constraint(Q2, xbar[i,j]<=Pmax*xbar[i,j+I]/swi)
            end
            for j=2I+2:J    
                @constraint(Q2, xbar[i,j]<=swi)
            end
            @constraint(Q2,(A*xbar[i,:]).<=-Ptdes1[i,:])

        end

        for i=1:k

            for j=1:3I+1
                @constraint(Q2, c[i,j]==cc[j])  
            end
    
            
            @constraint(Q2,(c[i,:]+epsilon*xbar[i,:]+((cprime[1,:,:]*u[i]+cprime[2,:,:]*u[i]^2+cprime[3,:,:]*u[i]^3+cprime[4,:,:])'*mu[i,I+2:I+1+V])+A5'*mu[i,1:I+1]+A4'*lam[i,:]).<=ck1[i,:])
            @constraint(Q2,-(c[i,:]+epsilon*xbar[i,:]+((cprime[1,:,:]*u[i]+cprime[2,:,:]*u[i]^2+cprime[3,:,:]*u[i]^3+cprime[4,:,:])'*mu[i,I+2:I+1+V])+A5'*mu[i,1:I+1]+A4'*lam[i,:]).<=ck1[i,:])
           

            @constraint(Q2,(((cprime[1,:,:]*u[i]+cprime[2,:,:]*u[i]^2+cprime[3,:,:]*u[i]^3+cprime[4,:,:])*xbar[i,:]).-(bb)).<=ck3[i,I+2:I+1+V])
            @constraint(Q2,ck3[i,I+2:I+1+V].>=0)

            @constraint(Q2,(A5*xbar[i,:].-Emax).<=ck3[i,1:I+1])
            @constraint(Q2,ck3[i,1:I+1].>=0)
        
            @constraint(Q2,(A4*xbar[i,:].-Eo1[i]).<=ck4[i,:])
            @constraint(Q2,-(A4*xbar[i,:].-Eo1[i]).<=ck4[i,:])
            


            for t=1:V
                @constraint(Q2, (mu[i,I+1+t]*((xbar[i,:]'*(cprime[1,t,:]*u[i]+cprime[2,t,:]*u[i]^2+cprime[3,t,:]*u[i]^3+cprime[4,t,:])).-bb)).<=ck2[i,I+1+t])
                @constraint(Q2, -(mu[i,I+1+t]*((xbar[i,:]'*(cprime[1,t,:]*u[i]+cprime[2,t,:]*u[i]^2+cprime[3,t,:]*u[i]^3+cprime[4,t,:])).-bb)).<=ck2[i,I+1+t])
            end
            for j=1:I+1
                @constraint(Q2,(mu[i,j]*(A5[j,:]'*xbar[i,:].-Emax)).<=ck2[i,j])  
                @constraint(Q2,-(mu[i,j]*(A5[j,:]'*xbar[i,:].-Emax)).<=ck2[i,j])  
            end


            for j=1:J
                @constraint(Q2,-(xbar[i,j]-x[i,j])<=p[i,j])
                @constraint(Q2,(xbar[i,j]-x[i,j])<=p[i,j])
            end

        end

        @objective(Q2, Min, sum(p)+sum(ff1.*ck1)+sum(ff2.*ck2)+sum(ff3.*ck3)+sum(ff4.*ck4)+10^2*(sum((xbar.-xprev).*(xbar.-xprev))))  # objective contains the l1-norm of decision varibale loss + total penalty  + proximal term
        optimize!(Q2)
        objv=getobjectivevalue(Q2)
        
        return value.(p),value.(xbar),value.(objv)
        
    end

    function dual_block(xbar,cprime,x,prevlam,prevmu) ## solving for the dual variables set while keeping the decision variables set and surrogate parameters set fixed
        Q3= Model(optimizer_with_attributes(Gurobi.Optimizer))
        @variables(Q3, begin
        lam[1:k,1:I]
        mu[1:k,1:I+1+V]>=0
        ck2[1:k,1:I+1+V]>=0  # penalty for complemnentary conditions
        ck1[1:k,1:J]>=0   # penalty for stationarity condition
        ck3[1:k,1:I+1+V]  # penalty for primal feasibility inequalities 
        ck4[1:k,1:I]>=0   # penalty for primal feasibility equalities
        p[1:k,1:3I+1]>=0
        c[1:k,1:J]
        end)
            

        for i=1:k

            for j=1:3I+1
                @constraint(Q3, c[i,j]==cc[j])  
            end
    
            
            @constraint(Q3,(c[i,:]+epsilon*xbar[i,:]+((cprime[1,:,:]*u[i]+cprime[2,:,:]*u[i]^2+cprime[3,:,:]*u[i]^3+cprime[4,:,:])'*mu[i,I+2:I+1+V])+A5'*mu[i,1:I+1]+A4'*lam[i,:]).<=ck1[i,:])
            @constraint(Q3,-(c[i,:]+epsilon*xbar[i,:]+((cprime[1,:,:]*u[i]+cprime[2,:,:]*u[i]^2+cprime[3,:,:]*u[i]^3+cprime[4,:,:])'*mu[i,I+2:I+1+V])+A5'*mu[i,1:I+1]+A4'*lam[i,:]).<=ck1[i,:])
           

            @constraint(Q3,(((cprime[1,:,:]*u[i]+cprime[2,:,:]*u[i]^2+cprime[3,:,:]*u[i]^3+cprime[4,:,:])*xbar[i,:]).-(bb)).<=ck3[i,I+2:I+1+V])
            @constraint(Q3,ck3[i,I+2:I+1+V].>=0)

            @constraint(Q3,(A5*xbar[i,:].-Emax).<=ck3[i,1:I+1])
            @constraint(Q3,ck3[i,1:I+1].>=0)
        
            @constraint(Q3,(A4*xbar[i,:].-Eo1[i]).<=ck4[i,:])
            @constraint(Q3,-(A4*xbar[i,:].-Eo1[i]).<=ck4[i,:])
            


            for t=1:V
                @constraint(Q3, (mu[i,I+1+t]*((xbar[i,:]'*(cprime[1,t,:]*u[i]+cprime[2,t,:]*u[i]^2+cprime[3,t,:]*u[i]^3+cprime[4,t,:])).-bb)).<=ck2[i,I+1+t])
                @constraint(Q3, -(mu[i,I+1+t]*((xbar[i,:]'*(cprime[1,t,:]*u[i]+cprime[2,t,:]*u[i]^2+cprime[3,t,:]*u[i]^3+cprime[4,t,:])).-bb)).<=ck2[i,I+1+t])
            end
            for j=1:I+1
                @constraint(Q3,(mu[i,j]*(A5[j,:]'*xbar[i,:].-Emax)).<=ck2[i,j])  
                @constraint(Q3,-(mu[i,j]*(A5[j,:]'*xbar[i,:].-Emax)).<=ck2[i,j])  
            end


            for j=1:J
                @constraint(Q3,-(xbar[i,j]-x[i,j])<=p[i,j])
                @constraint(Q3,(xbar[i,j]-x[i,j])<=p[i,j])
            end

        end

        @objective(Q3, Min, sum(p)+sum(ff1.*ck1)+sum(ff2.*ck2)+sum(ff3.*ck3)+sum(ff4.*ck4)+10^-2*(sum((mu.-prevmu).*(mu.-prevmu)))+10^-2*(sum((lam.-prevlam).*(lam.-prevlam)))) 
        optimize!(Q3)
        objv=getobjectivevalue(Q3)
        
        return value.(p),value.(lam),value.(mu),value.(objv)
        
    end
   
    function surr_param_block(xbar,lam,mu,x,prevcp) ## solving for the surrogate parameters set while keeping the dual variables set and decision variables set fixed
        Q4= Model(optimizer_with_attributes(Gurobi.Optimizer))
        set_optimizer_attribute(Q4, "BarConvTol", 1e-2)
        @variables(Q4, begin
        -bound<=cprime[1:4,1:V,1:J]<=bound
        spar[1:4,1:V,1:J]>=0
        ck2[1:k,1:I+1+V]>=0  # penalty for complemnentary conditions
        ck1[1:k,1:J]>=0   # penalty for stationarity condition
        ck3[1:k,1:I+1+V]  # penalty for primal feasibility inequalities 
        ck4[1:k,1:I]>=0   # penalty for primal feasibility equalities
        p[1:k,1:3I+1]>=0
        c[1:k,1:J]
        end)
        set_start_value.(cprime,prevcp)
       
    
        for i=1:k

            for j=1:3I+1
                @constraint(Q4, c[i,j]==cc[j])  
            end
    
            
            @constraint(Q4,(c[i,:]+epsilon*xbar[i,:]+((cprime[1,:,:]*u[i]+cprime[2,:,:]*u[i]^2+cprime[3,:,:]*u[i]^3+cprime[4,:,:])'*mu[i,I+2:I+1+V])+A5'*mu[i,1:I+1]+A4'*lam[i,:]).<=ck1[i,:])
            @constraint(Q4,-(c[i,:]+epsilon*xbar[i,:]+((cprime[1,:,:]*u[i]+cprime[2,:,:]*u[i]^2+cprime[3,:,:]*u[i]^3+cprime[4,:,:])'*mu[i,I+2:I+1+V])+A5'*mu[i,1:I+1]+A4'*lam[i,:]).<=ck1[i,:])
           

            @constraint(Q4,(((cprime[1,:,:]*u[i]+cprime[2,:,:]*u[i]^2+cprime[3,:,:]*u[i]^3+cprime[4,:,:])*xbar[i,:]).-(bb)).<=ck3[i,I+2:I+1+V])
            @constraint(Q4,ck3[i,I+2:I+1+V].>=0)

            @constraint(Q4,(A5*xbar[i,:].-Emax).<=ck3[i,1:I+1])
            @constraint(Q4,ck3[i,1:I+1].>=0)
        
            @constraint(Q4,(A4*xbar[i,:].-Eo1[i]).<=ck4[i,:])
            @constraint(Q4,-(A4*xbar[i,:].-Eo1[i]).<=ck4[i,:])
            


            for t=1:V
                @constraint(Q4, (mu[i,I+1+t]*((xbar[i,:]'*(cprime[1,t,:]*u[i]+cprime[2,t,:]*u[i]^2+cprime[3,t,:]*u[i]^3+cprime[4,t,:])).-bb)).<=ck2[i,I+1+t])
                @constraint(Q4, -(mu[i,I+1+t]*((xbar[i,:]'*(cprime[1,t,:]*u[i]+cprime[2,t,:]*u[i]^2+cprime[3,t,:]*u[i]^3+cprime[4,t,:])).-bb)).<=ck2[i,I+1+t])
            end
            for j=1:I+1
                @constraint(Q4,(mu[i,j]*(A5[j,:]'*xbar[i,:].-Emax)).<=ck2[i,j])  
                @constraint(Q4,-(mu[i,j]*(A5[j,:]'*xbar[i,:].-Emax)).<=ck2[i,j])  
            end


            for j=1:J
                @constraint(Q4,-(xbar[i,j]-x[i,j])<=p[i,j])
                @constraint(Q4,(xbar[i,j]-x[i,j])<=p[i,j])
            end

        end

        

        @objective(Q4, Min, sum(p)+sum(ff1.*ck1)+sum(ff2.*ck2)+sum(ff3.*ck3)+sum(ff4.*ck4)+10^-2*(sum((cprime.-prevcp).*(cprime.-prevcp))))
        optimize!(Q4)
        objv=getobjectivevalue(Q4)
        
        return value.(p),value.(cprime),value.(objv),value.(ck1),value.(ck2),value.(ck3),value.(ck4)
        
    end  

   niter=4
   global pp=zeros(50,niter)

   global lam_0=ones(k,I)
   global mu_0=ones(k,I+1+V)
   global cp_0=ones(4,V,J)
   global cp1_0=ones(4,V)


   global pp1=zeros(k,J)
   global ixbar=zeros(niter,k,J)
   global obj1

   global pp2=zeros(k,J)
   global ilam=zeros(niter,k,I)
   global imu=zeros(niter,k,I+V+1)
   global obj2

   global pp3=zeros(k,J)
   global icprime=zeros(niter,4,V,J)
   global icprime1=zeros(niter,4,V)
   global obj3
 
   global p1=zeros(k,J)
   global p2=zeros(k,I+V+1)
   global p3=zeros(k,I+V+1)
   global p4=zeros(k,I)


    cp_init=zeros(4,1,J)
    cp_init,lam_0,mu_0=initialization2(xnaught)
    x_0=xnaught
    
        for t=1:V
            cp_0[:,t,:]=cp_init
        end

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
                pp1,ixbar[iter,:,:],obj1=decision_block(lam_0,mu_0,cp_0,xnaught,x_0)
                pp2,ilam[iter,:,:],imu[iter,:,:],obj2=dual_block(ixbar[iter,:,:],cp_0,xnaught,lam_0,mu_0)
                pp3,icprime[iter,:,:,:],obj3,p1,p2,p3,p4=surr_param_block(ixbar[iter,:,:],ilam[iter,:,:],imu[iter,:,:],xnaught,cp_0)
                iter+=1
            else
                pp1,ixbar[iter,:,:],obj1=decision_block(ilam[iter-1,:,:],imu[iter-1,:,:],icprime[iter-1,:,:,:],xnaught,ixbar[iter-1,:,:])
                pp2,ilam[iter,:,:],imu[iter,:,:],obj2=dual_block(ixbar[iter,:,:],icprime[iter-1,:,:,:],xnaught,ilam[iter-1,:,:],imu[iter-1,:,:])
                println("OITER", oiter)
                pp3,icprime[iter,:,:,:],obj3,p1,p2,p3,p4=surr_param_block(ixbar[iter,:,:],ilam[iter,:,:],imu[iter,:,:],xnaught,icprime[iter-1,:,:,:])
                pp[oiter,iter-1]=(sum(abs.(ixbar[iter,:,:].-ixbar[iter-1,:,:]))+sum(abs.(ilam[iter,:,:].-ilam[iter-1,:,:]))+sum(abs.(imu[iter,:,:].-imu[iter-1,:,:]))+sum(abs.(icprime[iter,:,:,:].-icprime[iter-1,:,:,:])))

                    if (pp[oiter,iter-1]<=1 || iter==niter)
                        cp_0=icprime[iter,:,:,:]
                        cp1_0=icprime1[iter,:,:]
                        lam_0=ilam[iter,:,:]
                        mu_0=imu[iter,:,:]
                        break
                    else
                        iter+=1
                    end
            end
        end
        penalty[oiter]=sum(p1)+sum(p2)+sum(p3)+sum(p4)

        ggw=5  
        if (penalty[oiter]<=0.01 || oiter==4)  
          break
        else
            for i=1:k
                for j=1:J
                    if (p1[i,j]>0.1)
                     ff1[i,j]=ff1[i,j]+(p1[i,j])^0.1*ggw
                    else
                     ff1[i,j]=ff1[i,j]+(abs(p1[i,j]))^0.1*1
                    end
                end
            end
            for i=1:k
                for j=1:I+V+1
                    if (p2[i,j]>0.1)
                        ff2[i,j]=ff2[i,j]+(p2[i,j])^0.1*ggw
                    else
                        ff2[i,j]=ff2[i,j]+(abs(p2[i,j]))^0.1*1
                    end
                end
            end
            for i=1:k
                for j=1:I+V+1
                    if (p3[i,j]>0.1)
                        ff3[i,j]=ff3[i,j]+(p3[i,j])^0.1*ggw
                    else
                        ff3[i,j]=ff3[i,j]+(abs(p3[i,j]))^0.1*1
                    end
                end
            end
            for i=1:k
                for j=1:I
                    if (p4[i,j]>0.1)
                        ff4[i,j]=ff4[i,j]+(p4[i,j])^0.1*ggw
                    else
                        ff4[i,j]=ff4[i,j]+(abs(p4[i,j]))^0.1*1
                    end
                end
            end
        end

        oiter+=1
    end


A3=zeros(k,V,J)
b3=zeros(k,V)

for i=1:k
    for t=1:V
        for j=1:J
            A3[i,t,j]=cp_0[1,t,j]*u[i]+cp_0[2,t,j]*u[i]^2+cp_0[3,t,j]*u[i]^3+cp_0[4,t,j] 
         end
        b3[i,t]=cp1_0[1,t]*u[i]+cp1_0[2,t]*u[i]^2+cp1_0[3,t]*u[i]^3+cp1_0[4,t]
    end

end


X=zeros(k,J)
OV=zeros(k)
cx=zeros(k)

global dtt1=zeros(100)
for i=1:k
    Q9= Model(optimizer_with_attributes(Gurobi.Optimizer))
    @variable(Q9, xhat[1:J]>=0)   

    for j=1:I+1
        @constraint(Q9, xhat[j]>=0)
    end
    for j=2*I+2:3*I+1
        @constraint(Q9, swi>=xhat[j]>=0)
    end
    for j=I+2:2*I+1
       @constraint(Q9,  xhat[j]<=xhat[j+I]*(Pmax/swi))
    end

    @constraint(Q9, A*xhat.<=-Ptdes1[i,:])
    @constraint(Q9,A3[i,:,:]*xhat.<=bb) 
    @constraint(Q9,A4*xhat.==Eo1[i])
    @constraint(Q9,A5*xhat.<=Emax)

    @objective(Q9, Min, sum(cc[i]*xhat[i] for i=1:3I+1)+epsilon*(sum(xhat.*xhat))/2)

    optimize!(Q9)
    primal_status(Q9)
    println(primal_status(Q9))
    if (primal_status(Q9)==NO_SOLUTION)
        
        X[i,:]=zeros(J)
        OV[i]=0
    else
        
        X[i,:]=value.(xhat)
        OV[i]=objective_value(Q9)

    end
end

X[:,2I+2:3I+1]=round.(X[:,2I+2:3I+1])
global dtt2=zeros(100)
for i=1:k
    Qin= Model(optimizer_with_attributes(Gurobi.Optimizer))
        @variables(Qin, begin
        Ptbatin[1:I]
        Pmax>=Ptengin[1:I]>=0
        Emax>=Ein[1:I+1]>=0
        swi>=zin[1:I]>=0
        end)
        @constraint(Qin, zin.==round.(X[i,2I+2:3I+1]))
        @constraint(Qin,Ein[1]==Eo1[1])
            for j=1:I
                @constraint(Qin, Ein[j+1]==Ein[j]-tou*Ptbatin[j])
                @constraint(Qin, Ptbatin[j]+Ptengin[j]>=Ptdes1[i,j])
                @constraint(Qin, Ptengin[j]<=zin[j]*Pmax/swi)
            end

        @objective(Qin, Min, sum(cc[i]*Ein[i] for i=1:I+1)+sum(cc[i+I+1]*Ptengin[i] for i=1:I)+sum(cc[i+2I+1]*zin[i] for i=1:I))
        
        optimize!(Qin)
        primal_status(Qin)
    if (primal_status(Qin)==NO_SOLUTION)
            
            X[i,:]=zeros(J)
            OV[i]=0
    else

        for j=1:I+1
            X[i,j]=value.(Ein[j])
        end
        for j=I+2:2*I+1
            X[i,j]=value.(Ptengin[j-(I+1)])
        end
       
        OV[i]=objective_value(Qin)
    end
  
end



Atest=zeros(size(xtest,1),V,J)
btest=zeros(size(xtest,1),V)
OVt=zeros(size(xtest,1))
cxt=zeros(size(xtest,1))
Xtest=zeros(size(xtest,1),J)

for k1=1:size(xtest,1)
        for t=1:V
            for j=1:J
                
                Atest[k1,t,j]=cp_0[1,t,j]*utest[k1]+cp_0[2,t,j]*utest[k1]^2+cp_0[3,t,j]*utest[k1]^3+cp_0[4,t,j] 

            end
              btest[k1,t]=cp1_0[1,t]*utest[k1]+cp1_0[2,t]*utest[k1]^2+cp1_0[3,t]*utest[k1]^3+cp1_0[4,t]
        end

end

for k1=1:size(xtest,1)
    Q10= Model(optimizer_with_attributes(Gurobi.Optimizer))
    @variable(Q10, xprime[1:J]>=0)   

    for j=1:I+1
        @constraint(Q10, xprime[j]>=0) 
    end
    for j=2*I+2:3*I+1
        @constraint(Q10, swi>=xprime[j]>=0)
    end
    for j=I+2:2*I+1
        @constraint(Q10,  xprime[j]<=xprime[j+I]*(Pmax/swi))
    end

    @constraint(Q10, A*xprime.<=-Ptdestest[k1,:])
    @constraint(Q10,Atest[k1,:,:]*xprime.<=bb)   
    @constraint(Q10,A4*xprime.==Eo1[k1])
    @constraint(Q10,A5*xprime.<=Emax)

    @objective(Q10, Min,sum(cc[i]*xprime[i] for i=1:3I+1)+epsilon*(sum(xprime.*xprime))/2)

    optimize!(Q10)
    primal_status(Q10)
    println(primal_status(Q10))
    if (primal_status(Q10)==NO_SOLUTION)
        
        Xtest[k1,:]=zeros(J)
        OVt[k1]=0
    else
        
        Xtest[k1,:]=value.(xprime)
        OVt[k1]=objective_value(Q10)

    end

end


for k1=1:size(xtest,1)
    tt=time_ns()
    Qin= Model(optimizer_with_attributes(Gurobi.Optimizer))
        @variables(Qin, begin
        Ptbatin[1:I]
        Pmax>=Ptengin[1:I]>=0
        Emax>=Ein[1:I+1]>=0
        pp1[1:3I+1]>=0
        end)
        @variable(Qin, swi>=zin[1:I]>=0, Int)
        @constraint(Qin,Ein[1]==Eo1[1])
        for j=1:I
            @constraint(Qin, Ein[j+1]==Ein[j]-tou*Ptbatin[j])
            @constraint(Qin, Ptbatin[j]+Ptengin[j]>=Ptdestest[k1,j])
            @constraint(Qin, Ptengin[j]<=zin[j]*Pmax/swi)
        end

        for j=1:I+1
            @constraint(Qin, (Xtest[k1,j]-Ein[j])<=pp1[j])
            @constraint(Qin, -(Xtest[k1,j]-Ein[j])<=pp1[j])
        end

        for j=I+2:2I+1
            @constraint(Qin, (Xtest[k1,j]-Ptengin[j-(I+1)])<=pp1[j])
            @constraint(Qin, -(Xtest[k1,j]-Ptengin[j-(I+1)])<=pp1[j])
        end

        for j=2I+2:3I+1
            @constraint(Qin, (Xtest[k1,j]-zin[j-(2I+1)])<=pp1[j])
            @constraint(Qin, -(Xtest[k1,j]-zin[j-(2I+1)])<=pp1[j])
        end

        @objective(Qin, Min, sum(pp1[2I+2:3I+1]))
        
        optimize!(Qin)
        primal_status(Qin)
    if (primal_status(Qin)==NO_SOLUTION)
            
            Xtest[k1,:]=zeros(J)
            OVt[k1]=0
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
        OVt[k1]=sum(cc[i]* value.(Ein[i]) for i=1:I+1)+sum(cc[i+I+1]* value.(Ptengin[i]) for i=1:I)+sum(cc[i+2I+1]* value.(zin[i]) for i=1:I)
    end
    global dtt1[k1]=time_ns()-tt
end



for k1=1:size(xtest,1)
    tt=time_ns()
    Qin= Model(optimizer_with_attributes(Gurobi.Optimizer))
        @variables(Qin, begin
        Ptbatin[1:I]
        Pmax>=Ptengin[1:I]>=0
        Emax>=Ein[1:I+1]>=0
        end)
        @variable(Qin, swi>=zin[1:I]>=0, Int)
        @constraint(Qin, zin.==round.(Xtest[k1,2I+2:3I+1]))
        @constraint(Qin,Ein[1]==Eo1[1])
        for j=1:I
            @constraint(Qin, Ein[j+1]==Ein[j]-tou*Ptbatin[j])
            @constraint(Qin, Ptbatin[j]+Ptengin[j]>=Ptdestest[k1,j])
            @constraint(Qin, Ptengin[j]<=zin[j]*Pmax/swi)
        end

        @objective(Qin, Min, -cc[I+1]*(Emax-Ein[I+1])+sum(cc[i]*Ein[i] for i=1:I)+sum(cc[i+I+1]*Ptengin[i] for i=1:I)+sum(cc[i+2I+1]*zin[i] for i=1:I))

        optimize!(Qin)
        primal_status(Qin)
    if (primal_status(Qin)==NO_SOLUTION)
            
            Xtest[k1,:]=zeros(J)
            OVt[k1]=0
    else

        for j=1:I+1
            Xtest[k1,j]=value.(Ein[j])
        end
        for j=I+2:2*I+1
            Xtest[k1,j]=value.(Ptengin[j-(I+1)])
        end
        OVt[k1]=objective_value(Qin)
    end
    global dtt2[k1]=time_ns()-tt
end

sum1=zeros(size(X,1))
sum2=zeros(size(xtest,1))
osum1=zeros(size(X,1))
osum2=zeros(size(xtest,1))
sum3=zeros(size(Xtest,1))
osum3=zeros(size(Xtest,1))


global ctrain=0
global ctest=0
global clin=0

global X1=zeros(size(X))
global xnaught1=zeros(size(xnaught))
global Xtest1=zeros(size(Xtest))
global xtest1=zeros(size(xtest))

for i=1:size(X,1)
    for j=1:I+1
        X1[i,j]=X[i,j]/Eo1[1]
        xnaught1[i,j]=xnaught[i,j]/Eo1[1]
    end
    for j=I+2:3I+1
        X1[i,j]=X[i,j]
        xnaught1[i,j]=xnaught[i,j]
    end
end

for i=1:size(Xtest,1)
    for j=1:I+1    
        Xtest1[i,j]=Xtest[i,j]/Eo1[1]
        xtest1[i,j]=xtest[i,j]/Eo1[1]
    end
    for j=I+2:3I+1
        Xtest1[i,j]=Xtest[i,j]
        xtest1[i,j]=xtest[i,j]
    end
end





for i=1:(size(X,1))
    sum1[i]=100*(sum(abs.(X1[i,:]-xnaught1[i,:])))/(sum(abs.(xnaught1[i,:])))
    osum1[i]=100*abs(sum(cc[j]*xnaught[i,j] for j=1:3I+1)+epsilon*(sum(xnaught[i,:].*xnaught[i,:]))/2-OV[i])/abs(sum(cc[j]*xnaught[i,j] for j=1:3I+1)+epsilon*(sum(xnaught[i,:].*xnaught[i,:]))/2)
    if(X[i,1]==0)
        sum1[i]=0
        osum1[i]=0
        global ctrain+=1
    end

end

sumb=zeros(size(xtest,1))
sumc=zeros(size(xtest,1))

for i=1:(size(xtest,1))
    sum2[i]=100*(sum(abs.(Xtest1[i,:]-xtest1[i,:])))/(sum(abs.(xtest1[i,:])))
    osum2[i]=100*abs(sum(cc[j]*xtest[i,j] for j=1:3I+1)+epsilon*(sum(xtest[i,:].*xtest[i,:]))/2-cc[I+1]*(Emax)-OVt[i])/abs(sum(cc[j]*xtest[i,j] for j=1:3I+1)-cc[I+1]*(Emax)+epsilon*(sum(xtest[i,:].*xtest[i,:]))/2)
    sumc[i]=100*(sum(abs.(Xtest1[i,1:I+1]-xtest1[i,1:I+1])))/(sum(abs.(xtest1[i,1:I+1])))
    sumb[i]=100*(sum(abs.(Xtest1[i,2I+2:3I+1]-xtest1[i,2I+2:3I+1])))/(sum(abs.(xtest1[i,2I+2:3I+1])))
    
    if(Xtest[i,1]==0)
        sum2[i]=0
        osum2[i]=0
        global ctest+=1
    end
end

for i=1:(size(Xtest,1))
    sum3[i]=100*(sum(abs.(xin[i,:]-xtest1[i,:])))/(sum(abs.(xtest1[i,:])))
    osum3[i]=100*abs(sum(cc[j]*xtest[i,j] for j=1:3I+1)+epsilon*(sum(xtest[i,:].*xtest[i,:]))-OVin[i])/abs(sum(cc[j]*xtest[i,j] for j=1:3I+1)+epsilon*(sum(xtest[i,:].*xtest[i,:])))
    if(xin[i,1]==0)
        sum3[i]=0
        osum3[i]=0
        global clin+=1
    end


end

dtt=time()-init

if (k==3)
    str="w"
else
    str="a"
end
output_file=open("C:/UMN/Research/ICML/codes and data/MPC/result_1.jl",str)
write(output_file, "|K|= ") 
show(output_file,  k) 
write(output_file, "; \n \n")

write(output_file, "time= ") 
show(output_file,  dtt) 
write(output_file, "; \n \n")

write(output_file, "Trainin error= ") 
show(output_file,  sum(sum1)/(size(X,1)-ctrain )) 
write(output_file, "; \n \n")



write(output_file, "Test error (BV) = ") 
show(output_file, sum(sumb)/(size(xtest,1)-ctest)) 
write(output_file, "; \n \n")

write(output_file, "Test error (CV)= ") 
show(output_file, sum(sumc)/(size(xtest,1)-ctest)) 
write(output_file, "; \n \n")

write(output_file, "Test error= ") 
show(output_file, sum(sum2)/(size(xtest,1)-ctest)) 
write(output_file, "; \n \n")

write(output_file, "Objective Training error= ") 
show(output_file, sum(osum1)/(size(X,1)-ctrain )) 
write(output_file, "; \n \n")

write(output_file, "Objective Test error = ") 
show(output_file, sum(osum2)/(size(xtest,1)-ctest)) 
write(output_file, "; \n \n")

write(output_file, "Penalty ") 
show(output_file, penalty) 
write(output_file, "; \n \n")

write(output_file, "ctrain= ") 
show(output_file, ctrain) 
write(output_file, "; \n \n")

write(output_file, "ctest= ") 
show(output_file, ctest) 
write(output_file, "; \n \n \n \n")

write(output_file, "tim= ") 
show(output_file, (dtt1+dtt2)/10^9) 
write(output_file, "; \n \n \n \n")

write(output_file, "OVt= ") 
show(output_file, OVt) 
write(output_file, "; \n \n \n \n")
close(output_file)



output_file=open("C:/UMN/Research/ICML/codes and data/MPC/outputs_1.jl",str)
write(output_file, "sum2= ") 
show(output_file,  sum2) 
write(output_file, "; \n \n")

write(output_file, "osum2= ") 
show(output_file,  osum2) 
write(output_file, "; \n \n")

write(output_file, "Xtest= ") 
show(output_file,  Xtest) 
write(output_file, "; \n \n")

write(output_file, "xtest= ") 
show(output_file, xtest) 
write(output_file, "; \n \n")
close(output_file)
