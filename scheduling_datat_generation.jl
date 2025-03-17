using LinearAlgebra, Random, Gurobi, GAMS, DataFrames, CSV, Printf, JuMP, Ipopt;
using Distributed
using JuMP, Gurobi
using CPUTime, Statistics


S=700 # number of data points to be generated


### change these parameters as per problem size ###
I=20 # number of batches   
J=4  # number of units
eta=100 # Horizon length
T=10 # number of slots
V=2J*T+J # variables for each batch 


global uu=zeros(S) # input parameter value for each instance
global xijt=zeros(S,I,J,T) # binary assignment varibales 
global tjt=zeros(S,J,T) # continuous variables for the length of slots on each unit 
global sij=zeros(S,I,J) # continuous variables for the start time of a slot on a unit.

global dtt=zeros(S)

global touij=zeros(S,I,J) # processing time of a batch on a given unit 
global rhoi=zeros(S,I) # release time of the batch
global epsi=zeros(S,I) # due time of the batch
global gamijP=zeros(S,I,J,T) # Processing cost of the batch on a given unit


global obv=zeros(S) # array to store the optimal objective values
global counter=0
global c=1

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

global UT=0

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

for k=1:S
    global uu[k]=rand(100:900)/100
    for i=1:I
            global touij[k,i,:]=[(3sin(uu[k]*i)+4) (3cos(uu[k]*i)+4) (3sin(uu[k]*(i+100))+4) (40*sin(uu[k]*(i+100))+50)/10]/0.5 # 10 8 7
            touij[k,i,:]=round.(touij[k,i,:],digits=2)

        #global rhoi[k,i]=(8sin(uu[k]*(i+100))^2)+1
        global rhoi[k,i]=(30sin(uu[k]*(i+100))^2)+1
        rhoi[k,i]=round.(rhoi[k,i],digits=2)

        #global epsi[k,i]=rhoi[i]+(10sin(uu[k]*i/100)+20)
        global epsi[k,i]=rhoi[k,i]+(20sin(uu[k]*i)+40)
        epsi[k,i]=round.(epsi[k,i],digits=2)

        for j=1:J
            if (touij[k,i,j]<=0.1)
                global gamijP[k,i,j,:].=0
                global touij[k,i,j]=0
            else
                global gamijP[k,i,j,:].=(120-4*touij[k,i,j])
                gamijP[k,i,j,:]=round.(gamijP[k,i,j,:],digits=2)
            end
        end
        for j=1:J
            for t=1:T
               # if(gamijP[k,i,j,t]>0.5)
                    global gamijP[k,i,j,t]=gamijP[k,i,j,t]+(t-1)*2 #rand(-50:50)/10
               # end
            end
        end




        for j=1:J
            global A5[i,j,(j-1)*T+1:j*T].=-epsi[k,i]
            #global A5[i,j,(j-1)*T+1:j*T].=-100
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

    

    Tj=eta*ones(J) # horizon length for each unit

    #T=(maximum(ceil.(Int,mean(touij, dims=1).+std(touij, dims=1))))  # maybe used to provide a guesstimate of the number of slots

    tt=time()
        Q1= Model(optimizer_with_attributes(Gurobi.Optimizer))
        set_optimizer_attribute(Q1, "MIPGap", 0.01)
        set_optimizer_attribute(Q1, "TIME_LIMIT", 1000)

        
        @variable(Q1, Xijt[1:I,1:J,1:T], Bin )
        @variable(Q1, Sij[1:I,1:J]>=0)
        @variable(Q1, Tjt[1:J,1:T]>=0)
        @variable(Q1, xbar[1:I,1:V])

        for i=1:I
            for j=1:J
                @constraint(Q1, xbar[i,(j-1)*(T)+1:j*T].==Xijt[i,j,:])  # transforming variable matrix into a variable array to be used with constraint matrices
            end
                @constraint(Q1, xbar[i,J*T+1:J*T+J].==Sij[i,:])
            for j=1:J
                @constraint(Q1, xbar[i,(j-1)*(T)+(J*T+J)+1:j*T+(J*T+J)].==Tjt[j,:])
            end
        end

        for i=1:I
            for j=1:J
                if(touij[k,i,j]==0)
                    @constraint(Q1, Xijt[i,j,:].==0)
                end
            end
        end

        #=for t=1:T
            for j=1:J
                if (t>1)
                   @constraint(Q1, Tjt[j,t]>=Tjt[j,t-1])  # constraint in the original formulation
                end
            end
        end=#

        @constraint(Q1,A3*xbar[1,:].<=0) # same constraint in a matrix representation


        #=for j=1:J
            @constraint(Q1, Tjt[j,T]==Tj[j])
        end=#

        @constraint(Q1,A4*xbar[1,:].==Tj)
        


        for i=1:I
            #@constraint(Q1, sum(sum(Xijt[i,j,t] for j=1:J) for t=1:T)==1)
            @constraint(Q1, A1*xbar[i,:].==1)
        end
       

        for j=1:J*T
            #for t=1:T
                #@constraint(Q1, sum(Xijt[i,j,t] for i=1:I)<=1)
                @constraint(Q1, A2*xbar[:,j].<=1)
            #end
        end
    
        for i=1:I
            #=for j=1:J
                @constraint(Q1, Sij[i,j]<=epsi[k,i]*sum(Xijt[i,j,t] for t=1:T))
            end=#
            @constraint(Q1, A5[i,:,:]*xbar[i,:].<=0)
        end
       
        for i=1:I
            #=for j=1:J
                for t=1:T
                    if (t>1)
                       # @constraint(Q1, Sij[i,j]>=Tjt[j,t-1]-eta*(1-Xijt[i,j,t]))
                    else
                       # @constraint(Q1, Sij[i,j]>=-eta*(1-Xijt[i,j,t]))
                    end

                   # @constraint(Q1, Sij[i,j]+touij[k,i,j]<=Tjt[j,t]+eta*(1-Xijt[i,j,t]))
                end
            end=#

            for j=1:J
                @constraint(Q1, A8[j,:,:]*xbar[i,:].<=eta)
                @constraint(Q1, A9[j,:,:]*xbar[i,:].<=eta-touij[k,i,j])
            end

        end

        
    for i=1:I
            #@constraint(Q1, sum(Sij[i,j] for j=1:J)>=rhoi[k,i])
            @constraint(Q1, A6*xbar[i,:].>=rhoi[k,i])
            
            #@constraint(Q1, sum(Sij[i,j] for j=1:J)+sum(sum(touij[k,i,j]*Xijt[i,j,t] for t=1:T) for j=1:J)<=epsi[k,i])
            @constraint(Q1, A7[i,:,:]*xbar[i,:].<=epsi[k,i])
    end
    

   #=for j=1:J
        for t=2:T
            @constraint(Q1, sum(Xijt[i,j,t] for i=1:I)<=sum(Xijt[i,j,t-1] for i=1:I))  #symmetry breaking??
        end
    end=#

    for j=1:J*T
        if((j-1)%T!=0)
            #@constraint(Q1, sum(xbar[i,j] for i=1:I)<=sum(xbar[i,j-1] for i=1:I))
        end
    end
   
        @objective(Q1, Min, sum(sum(sum(gamijP[k,i,j,t]*Xijt[i,j,t] for i=1:I) for j=1:J) for t=1:T))
        optimize!(Q1)

    println(k)
    if (primal_status(Q1)==NO_SOLUTION)
        global counter=counter+1
        global obv[k]=0
        k=k+1

    else
        global obv[k]=getobjectivevalue(Q1)
        global dtt[k]=time()-tt
        if (dtt[k]>990)
            global UT
            UT=UT+1

        end
        global xijt[k,:,:,:]=value.(Xijt)
        global tjt[k,:,:]=value.(Tjt)
        global sij[k,:,:]=value.(Sij)
    end
    global UT
    if(UT>=30)
        break
    end


end

## finding unique solutions generated for S data points ##
global counter1=0
MM=unique((i->begin uu[i] end),1:S)
for i=1:lastindex(MM,1)
    if (obv[MM[i]]==0)
        global counter1=counter1+1  # discarding any generated infeasible instances
    end
end


global UU=zeros(Float64, size(MM,1)-counter1 )
global XIJT=zeros(Float64, size(MM,1)-counter1, I,J,T)
global SIJ=zeros(Float64, size(MM,1)-counter1,I,J)
global TJT=zeros(Float64, size(MM,1)-counter1,J,T)
global TOU=zeros(Float64, size(MM,1)-counter1,I,J)
global GAMI=zeros(Float64, size(MM,1)-counter1,I,J,T)
global GAMJ=zeros(Float64, size(MM,1)-counter1,J,T)
global GAMS1=zeros(Float64, size(MM,1)-counter1,I,J)
global RHO=zeros(Float64, size(MM,1)-counter1,I)
global EPS=zeros(Float64, size(MM,1)-counter1,I)
global OBV=zeros(Float64, size(MM,1)-counter1)
global TIM=zeros(Float64, size(MM,1)-counter1)


global cc=1
for i=1:lastindex(MM,1)
    if (obv[MM[i]]!=0)
        global cc
        global UU[cc]=uu[MM[i]]
        global XIJT[cc,:,:,:]=xijt[MM[i],:,:,:]
        global SIJ[cc,:,:]=sij[MM[i],:,:]
        global TJT[cc,:,:]=tjt[MM[i],:,:]
        global TOU[cc,:,:]=touij[MM[i],:,:]
        global GAMI[cc,:,:,:]=gamijP[MM[i],:,:,:]
        global RHO[cc,:]=rhoi[MM[i],:]
        global EPS[cc,:]=epsi[MM[i],:]
        global OBV[cc]=obv[MM[i]]
        global TIM[cc]=dtt[MM[i]]
        cc=cc+1
    end
    
end




output_file=open("C:/UMN/data_file2.jl","w") ## change destination
write(output_file, "u= ")
show(output_file, UU) 
write(output_file, "; \n \n")

write(output_file, "Xijt= ") 
show(output_file, XIJT) 
write(output_file, "; \n \n")

write(output_file, "Sij= ") 
show(output_file, SIJ) 
write(output_file, "; \n \n")

write(output_file, "Tjt= ") 
show(output_file, TJT) 
write(output_file, "; \n \n")

write(output_file, "touij= ") 
show(output_file, TOU) 
write(output_file, "; \n \n")

write(output_file, "gamijP= ") 
show(output_file, GAMI) 
write(output_file, "; \n \n")

write(output_file, "rhoi= ") 
show(output_file, RHO) 
write(output_file, "; \n \n")

write(output_file, "epsi= ") 
show(output_file, EPS) 
write(output_file, "; \n \n")

write(output_file, "obj= ") 
show(output_file, OBV) 
write(output_file, "; \n \n")

write(output_file, "time1= ") 
show(output_file, TIM) 
write(output_file, "; \n \n")


close(output_file)