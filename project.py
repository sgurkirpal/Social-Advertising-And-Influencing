import networkx as nx
import numpy
import matplotlib.pyplot as plt
import random
import collections

#package import
g=nx.read_edgelist("facebook_combined.txt",create_using=nx.Graph(),nodetype=int)

total_nodes=g.number_of_nodes()

#pagerank
rank=nx.pagerank(g)
sorted_rank={k:v for k,v in sorted(rank.items(),key=lambda item:item[1],reverse=True)}

threshold_input=float(input("Please enter the threshhold value "))
desired_reach=int(input("Please enter the reach you desire for your product(in %population) "))

#dictionaries
#dictionary to store degree of each node
degree_dict={}
for i in g.nodes():
    degree_dict[i]=nx.degree(g,i)
   
#neigh_dict gives list of all the neighbouring nodes
neigh_dict={}
for i in g.nodes():
    neigh_dict[i]=list(g.neighbors(i))

#functions
#hfun calculates the h_index of each node.
#we calculated the h-index by finding the degrees of immediate neighbors....
def h_fun(g,n):
    neigh=neigh_dict[n]
    degrees=[]
    for i in neigh:
        degrees.append(degree_dict[i])
    degrees.sort()
    sam=0
    for i in range(len(degrees)):
        if(degrees[i]>len(degrees)-i):
            sam=i
            break
    return len(degrees)-sam

#this dictionary assigns h-index value to each node
h_index_dict={}
for i in g.nodes:
    h_index_dict[i]=h_fun(g,i)


#clusterrank is a local centrality index in which we considered the degree of first 
#node and then multipliplied it by f(c)(which we calculated using outdegrees of each nodes)
def clusterrank(g):
    clustering_coefficient={}
    clustering_coefficient=nx.clustering(g)
    for i in clustering_coefficient:
        if(clustering_coefficient[i]!=0):
            clustering_coefficient[i]=10**(-0.5*clustering_coefficient[i])
    for i in clustering_coefficient:
        a=0
        for j in neigh_dict[i]:
            a+=degree_dict[j]+1
        clustering_coefficient[i]*=a
    return clustering_coefficient


#the local H-index is a semi-localcentrality measure, because it leverages the H-index 
#centrality to the second-order neighbors of a given node.
def lh_index(g,n):
    ans=0
    ans+=h_index_dict[n]
    neigh=neigh_dict[n]
    for i in neigh:
        ans+=h_index_dict[i]
    return ans


#this functions provides us the degree centrality of the node
#it is the simplest local centrality measure.
def deg_central(g,n):
    return degree_dict[n]


#The neighborhood connectivity of a vertex i is deﬁned as theaverage connectivity of 
#all neighbors of i
def neigh_connectivity(g,n):
    sumo=0
    neigh=neigh_dict[n]
    if(degree_dict[n]==0):
        return 0
    for i in neigh:
        sumo+=degree_dict[i]
    return sumo/degree_dict[n]


#Collective inﬂuence is a novel global centrality metric that measures the collective number 
#of nodes that can be reached from a given node i
def collective_inf(g,n):
    total=0
    for l in neigh_dict[n]:
        for j in neigh_dict[l]:
            #for k in neigh_dict[j]:
            total+=degree_dict[j]-1
    return ((total)*(degree_dict[n]-1))


#initializations
hello={}    #for h-index
lh={}       #for lh-index
DC={}       #degree centrality
BC={}       #betweenness centrality
NC={}       #neighborhood connectivity
CR={}       #clusterrank
CI={}       #collective influence


#assigning values from functions
for i in g.nodes:
    hello[i]=h_fun(g,i)
    lh[i]=lh_index(g, i)
    DC[i]=deg_central(g, i)
    NC[i]=neigh_connectivity(g, i)  
    CI[i]=collective_inf(g,i)
BC=nx.betweenness_centrality(g,k=100)
CR=clusterrank(g)
   
# Spreading score,is reﬂective of the potential of vertices in spreading of informa-tion
# within a network.
#spreading to get ivi
def spreading(g,n):
    return (NC[n]+CR[n])*(BC[n]+CI[n])


#the additive product of local Hindex and de-gree centrality is the Hubness score, 
#which could be reﬂective of the sovereignty of a vertex in it surrounding local territory
def hubness(g,n):
    return DC[n]+lh[n]


# IVI is the synergistic product of the most important local (i.e., degree centrality 
#and Cluster-Rank), semi-local (i.e., neighborhood connectivity and local Hin-dex), 
#and global (i.e., betweenness centrality and collective inﬂu-ence) centrality measures 
#in a way that simultaneously removespositional biases
def ivi(g,n):
    return spreading_dict[n]*hubness_dict[n]


#storing ivi values
ivi_dict={}
spreading_dict={}
hubness_dict={}
for i in list(g.nodes):
    spreading_dict[i]=spreading(g,i)
    hubness_dict[i]=hubness(g,i)
    ivi_dict[i]=ivi(g,i)
   
   
#sorted values in decreasing order.
sorted_hello={k:v for k,v in sorted(hello.items(),key=lambda item:item[1],reverse=True)}
sorted_lh={k:v for k,v in sorted(lh.items(),key=lambda item:item[1],reverse=True)}
sorted_dc={k:v for k,v in sorted(DC.items(),key=lambda item:item[1],reverse=True)}
sorted_nc={k:v for k,v in sorted(NC.items(),key=lambda item:item[1],reverse=True)}
sorted_bc={k:v for k,v in sorted(BC.items(),key=lambda item:item[1],reverse=True)}
sorted_cr={k:v for k,v in sorted(CR.items(),key=lambda item:item[1],reverse=True)}
sorted_ci={k:v for k,v in sorted(CI.items(),key=lambda item:item[1],reverse=True)}
sorted_ivi={k:v for k,v in sorted(ivi_dict.items(),key=lambda item:item[1],reverse=True)}


#printing top 10 influential nodes through various methods
'''
***************************************************************
PRINTING
***************************************************************
'''
cnt=0
for i in sorted_rank:
    print("pagerank ",i,sorted_rank[i])
    cnt+=1;
    if(cnt==10):
        break
   
cnt=0
for i in sorted_nc:
    print("neigh connectivity ",i,sorted_nc[i])
    cnt+=1;
    if(cnt==10):
        break
   
cnt=0
for i in sorted_lh:
    print("lh ",i,sorted_lh[i])
    cnt+=1;
    if(cnt==10):
        break
   
cnt=0
for i in sorted_dc:
    print("degree centrality ",i,sorted_dc[i])
    cnt+=1;
    if(cnt==10):
        break
   
cnt=0
for i in sorted_cr:
    print("clusterrank ",i,sorted_cr[i])
    cnt+=1;
    if(cnt==10):
        break  

cnt=0
for i in sorted_bc:
    print("betweenness ",i,sorted_bc[i])
    cnt+=1;
    if(cnt==10):
        break

cnt=0
for i in sorted_ci:
    print("collective influece ",i,sorted_ci[i])
    cnt+=1;
    if(cnt==10):
        break
   

cnt=0
for i in sorted_ivi:
    print("ivi ",i,sorted_ivi[i])
    cnt+=1;
    if(cnt==10):
        break
   
   
'''
***************************************************************
PRINTING OVER
***************************************************************
'''
   
def thres_value(g,n,l):
    b=neigh_dict[n]
    a=0
    for i in b:
        if(l[i]==1):
            a+=1
    return a/len(b)

sorted_ivi_keys=list(sorted_ivi.keys())
sorted_dc_keys=list(sorted_dc.keys())
sorted_lh_keys=list(sorted_lh.keys())
sorted_bc_keys=list(sorted_bc.keys())
sorted_cr_keys=list(sorted_cr.keys())
sorted_ci_keys=list(sorted_ci.keys())
sorted_nc_keys=list(sorted_nc.keys())
normal_list=[]
normal_list=list(g.nodes())
random.shuffle(normal_list)

def threshold(threshold_input,g,sorted_ivi_keys):
     visited_check=[]
     count=0
     for i in range(len(list(g.nodes()))):
         visited_check.append(0)
     node_count=0
     sorted_ivi_index=0
     my_queue=[sorted_ivi_keys[0]]
     count+=1
     visited_check[sorted_ivi_keys[0]]=1
     while(len(set(visited_check))==2):
         if(node_count>=len(my_queue)):
             sorted_ivi_index+=1
             if(visited_check[sorted_ivi_keys[sorted_ivi_index]]==1):
                 continue
             else:
                 count+=1
                 my_queue.append(sorted_ivi_keys[sorted_ivi_index])
                 visited_check[sorted_ivi_keys[sorted_ivi_index]]=1
         a=neigh_dict[my_queue[node_count]]
         node_count+=1
         for i in a:
             if(visited_check[i]==1):
                 continue
             if(thres_value(g,i,visited_check)>=threshold_input):
                 visited_check[i]=1
                 my_queue.append(i)
     return count
 
    
#uncomment this to draw the graphs.
'''
***************************************************************
GRAPH
***************************************************************
'''
#uncomment this to draw the graphs.

'''
x=[]
y=[]
for i in range(1,101):
    x.append(i/100)
    y.append(threshold(i/100,g,normal_list))
plt.plot(x,y,color='olive',alpha=0.9,label='Random')


x=[]
y=[]
for i in range(1,101):
    x.append(i/100)
    y.append(threshold(i/100,g,sorted_dc_keys))
plt.plot(x,y,color='green',alpha=0.9,label='Degree Centrality')
x=[]
y=[]
for i in range(1,101):
    x.append(i/100)
    y.append(threshold(i/100,g,sorted_bc_keys))
plt.plot(x,y,color='black',alpha=0.9,label='Betweenness Centrality')
x=[]
y=[]
for i in range(1,101):
    x.append(i/100)
    y.append(threshold(i/100,g,sorted_lh_keys))
plt.plot(x,y,color='red',alpha=0.9,label='Local H-index')

x=[]
y=[]
for i in range(1,101):
    x.append(i/100)
    y.append(threshold(i/100,g,sorted_nc_keys))
plt.plot(x,y,color='orange',alpha=0.9,label='Neighborhood Connectivity')
   
x=[]
y=[]
for i in range(1,101):
    x.append(i/100)
    y.append(threshold(i/100,g,sorted_cr_keys))
plt.plot(x,y,color='teal',alpha=0.9,label='ClusterRank')

x=[]
y=[]
for i in range(1,101):
    x.append(i/100)
    y.append(threshold(i/100,g,sorted_ci_keys))
plt.plot(x,y,color='purple',alpha=0.9,label='Collective Influence')

x=[]
y=[]
graphs=[]
for i in range(1,101):
    x.append(i/100)
    y.append(threshold(i/100,g,sorted_ivi_keys))
plt.plot(x,y,color='blue',alpha=0.9,label='IVI')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.show()
'''
'''
 
 
***************************************************************
GRAPH
***************************************************************
'''  

#this is profit list obtained from different threshold input values

threshold_input_list=[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]
#we have obtained this list(profit list) after running the code for above threshold input values 
#It took 2-3 hours to run so it is not included in the code now.
profit_list=[
0,
0,
19.104835046864032,
13.897165321746327,
14.757623785817161,
19.2836571189192,
21.234596353416464,
21.06628530899223,
23.806502878724633,
23.054959246384968,
26.619508064808993,
25.64558236592735,
24.78673120158613,
24.90229021554957,
22.297218519018926,
19.824915177077035,
16.42558727213464,
12.305310040529134
]
#uncomment this to make graph
#plt.plot(threshold_input_list,profit_list)


#minimizing cost and maximizing profit
#everything has been done using the optimized ivi results
visited_or_not=[0 for i in range(total_nodes)]   #keep track of nodes that have already been influenced


#gives weight/importance of a node at a given instant
def get_weight(n,threshold_input):
    if(visited_or_not[n]==1):
        return 0
    dummm=visited_or_not
    weigh=0
    weigh+=fee_dict[n]
    cnt=0
    my_queue=[n]
    while(cnt<len(my_queue)):
        dummy_list=neigh_dict[my_queue[cnt]]
        for i in dummy_list:
            if(dummm[i]==0):
                if(thres_value(g,i,visited_or_not)>=threshold_input):
                    dummm[i]=1
                    weigh+=fee_dict[i]
                    my_queue.append(i)
        cnt+=1
    return weigh


#gives the number of nodes influenced because of a particular node
def affected_nodes(n,threshold_input,influenced_peeps):
    cnt=0
    my_queue=[n]
    while(cnt<len(my_queue)):
        dummy_list=neigh_dict[my_queue[cnt]]
        for i in dummy_list:
            if(visited_or_not[i]==0):
                if(thres_value(g,i,visited_or_not)>=threshold_input):
                    visited_or_not[i]=1
                    influenced_peeps+=1
                    my_queue.append(i)
        cnt+=1
    return influenced_peeps


fee_dict={}         #fees of nodes to advertise them
weight_dict={}      #weight/importance of nodes
cnt=0
influenced_peeps=1
for i in sorted_ivi:
    fee_dict[i]=total_nodes-cnt
    cnt+=1
total_cost=0
for i in g.nodes():
    weight_dict[i]=get_weight(i,threshold_input)
knapsack_dict={}
for i in fee_dict:
    knapsack_dict[i]=weight_dict[i]/fee_dict[i]
max_knapsack=max(knapsack_dict, key=knapsack_dict.get)  #selecting maximum knapsack
visited_or_not[max_knapsack]=1
influenced_peeps=affected_nodes(max_knapsack,threshold_input,influenced_peeps)
total_cost+=fee_dict[max_knapsack]

bbb=[max_knapsack]

while(len(set(visited_or_not))!=1):
    if(influenced_peeps<desired_reach*total_nodes/100):
        for i in g.nodes():
            weight_dict[i]=get_weight(i,threshold_input)
        for i in fee_dict:
            knapsack_dict[i]=weight_dict[i]/fee_dict[i]
       
        max_knapsack=max(knapsack_dict, key=knapsack_dict.get)
        visited_or_not[max_knapsack]=1
        
        influenced_peeps=affected_nodes(max_knapsack,threshold_input,influenced_peeps)
        bbb.append(max_knapsack)
        total_cost+=fee_dict[max_knapsack]
    else:
        break

    
def threshold_again(threshold_input,g,sorted_ivi_keys,given_inf):
     to_inf=0
     visited_check=[]
     count=0
     for i in range(len(list(g.nodes()))):
         visited_check.append(0)
     node_count=0
     sorted_ivi_index=0
     my_queue=[sorted_ivi_keys[0]]
     count+=1
     visited_check[sorted_ivi_keys[0]]=1
     to_inf+=1
     while(len(set(visited_check))==2):
         if to_inf<given_inf:
             if(node_count>=len(my_queue)):
                 sorted_ivi_index+=1
                 if(visited_check[sorted_ivi_keys[sorted_ivi_index]]==1):
                     continue
                 else:
                     count+=1
                     my_queue.append(sorted_ivi_keys[sorted_ivi_index])
                     visited_check[sorted_ivi_keys[sorted_ivi_index]]=1
                     to_inf+=1
             a=neigh_dict[my_queue[node_count]]
             node_count+=1
             for i in a:
                 if(visited_check[i]==1):
                     continue
                 if(thres_value(g,i,visited_check)>=threshold_input):
                     visited_check[i]=1
                     to_inf+=1
                     my_queue.append(i)
         else:
             break
     return count
 

total_money=0
total_coun=threshold_again(threshold_input,g,sorted_ivi_keys,desired_reach*total_nodes/100)
highest_price=total_nodes
tot=total_coun
while(total_coun!=0):
    total_money+=highest_price
    highest_price-=1
    total_coun-=1
money_saved=total_money-total_cost
    
to_advertise_for_all=threshold(threshold_input,g,sorted_ivi_keys)


    
print("You need to advertise atleast",to_advertise_for_all,"people for 100% reach")
print("You need to advertise atleast",tot,"people for",desired_reach,"% reach")
print("Money spent in a naive manner:",total_money)
print("Money spent after optimising the approach:",total_cost)
money_saved=total_money-total_cost
print("money saved:",money_saved)
print("profit % ",money_saved/total_money*100)
