#Importing needed libraries:
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from collections import defaultdict
import osmnx as ox
from folium.features import DivIcon
from geopy.geocoders import Nominatim
import folium
INF = float('inf')
#Here we are inputing the Addresses we need to geocode:
loc = [['Narela Delhi'],['Bawana Delhi'],['Alipur Delhi'],['Delhi Technological University Delhi'],['Jahangirpuri Delhi'],['Punjabi Bagh Delhi'],['North Campus Delhi']]

#(1)FOR GREEN ROUTE: Weighted Adjacent Matrix for the individual locations that we enetered For Pollution (matrix size should be same to number of addresses):
gmat = [[0, -2, -1, -2, 0, 0, 0],
         [-2, 0, 1, 1, 0, 0, 0],
         [-1, 1, 0, 0, -1, 0, 0],
         [-2, 1, 0, 0, -1, 0, 0],
         [0, 0, -1, -1, 0, 0, -1],
         [0, 0, 0, 0, 0, 0, -1],
         [0, 0, 0, 0, -1, -1, 0]]

#(2)FOR YELLOW ROUTE: Weighted Adjacent Matrix for the individual locations that we enetered For Shortest distance (matrix size should be same to number of addresses):
lmat = [[0, 11, 10, 17, 0, 0, 0],
         [11, 0, 10, 12, 0, 0, 0],
         [10, 10, 0, 11, 10, 0, 0],
         [17, 12, 11, 0, 8, 0, 0],
         [0, 0, 10, 8, 0, 10, 9],
         [0, 0, 0, 0, 10, 0, 10],
         [0, 0, 0, 0, 9, 10, 0]]

#(3)FOR BLUE ROUTE: Weighted Adjacent Matrix for the individual locations that we enetered For FastestTime (matrix size should be same to number of addresses):
tmat = [[0, 18, 17, 28, 0, 0, 0],
         [18, 0, 17, 20, 0, 0, 0],
         [17, 17, 0, 18, 17, 0, 0],
         [28, 20, 18, 0, 13, 0, 0],
         [0, 0, 17, 13, 0, 17, 15],
         [0, 0, 0, 0, 17, 0, 17],
         [0, 0, 0, 0, 15, 17, 0]]

matrices = [gmat,lmat,tmat]

#GEOCODING( Finding Coordinates ):
#Making an instance of Nominatim class:
geolocator = Nominatim(user_agent="my_request")

address = []
long=[]
lat=[]

#Applying geocode method to get the location Coordinates:
for i in range(len(loc)):
  location = geolocator.geocode(loc[i])
  address.append(location.address)
  long.append(location.longitude)
  lat.append(location.latitude)

#Printing address and coordinates and Getting final needed coordinates:
print("Coordinates of all the entered Locations:")
for i in range(len(loc)):
 print('Address: ', address[i],'\n','Latitude: ', lat[i],'\n', 'Longitude: ', long[i], '\n')

#Generating our osmnx map:
#Displaying Paths(Shortest path in blue and shortest path according to matrix in green) on Map CODE3:
#ox.config(log_console=True, use_cache=True)
ox.settings.log_console = True
ox.settings.use_cache = True
locator = Nominatim(user_agent = "myapp")

# Location where you want to find your route
place = 'Delhi, India'

#Mode of travel for Shortest Path:
mode = 'drive'       # 'drive', 'bike', 'walk'

#Shortest path based on distance or time:
optimizer = 'length'        # 'length','time'

graph = ox.graph_from_place(place, network_type=mode)
speed = ox.add_edge_speeds(graph)
# hwy_speeds={ 'motorway':25, 'residential':25, 'primary':25, 'secondary':25, 'tertiary':25, 'unclassified':25, 'motorway_link':25, 'primary_link':25, 'secondary_link':25, 'tertiary_link':25}
speed=ox.add_edge_travel_times(speed)


distances=[]
time_travels =[]
greennodes=[]
yellownodes=[]
bluenodes=[]
#Looping through our 3 maps:
for k in range(3):
    wmat = matrices[k]

    #For Green Matrix(k==0) if EDGES ENTERED ARE NEGATIVE, TO RESOLVE THE NEGATIVE CYCLE:
    if k == 0:
        #CHECKING IF NEG CYCLY EXISTS:
        tempmat = matrices[k]
        for i in range(len(tempmat)):
            for j in range(len(tempmat)):
                if i != j and tempmat[i][j] == 0:
                    tempmat[i][j] = INF

        N = len(matrices[k])
        def NegativeCycle(AdjacMatrix, NegCyc):
            # cost will store shortest-path information and At start cost would be equal to the weight of the edge:
            edgecosts = [[AdjacMatrix[v][u] for u in range(N)] for v in range(N)]
            for k in range(N):
                for v in range(N):
                    for u in range(N):
                        # If vertex k is located on shortest path from v to u, then we will update the value of cost[v][u]:
                        if (edgecosts[v][k] != INF and edgecosts[k][u] != INF and edgecosts[v][k] + edgecosts[k][u] <
                                edgecosts[v][u]):
                            edgecosts[v][u] = edgecosts[v][k] + edgecosts[k][u]
                        # The graph contains a negative weight cycle, if the diagonal elements become negative:
            if edgecosts[v][v] < 0:  # Cost of diagonal elements
                print("NEGATIVE CYCLE DETECTED")

                NegCyc = 1
            else:
                print("NO NEGATIVE CYCLE DETECTED")
                NegCyc = 0

            return NegCyc


        val = NegativeCycle(tempmat, 0)  # Calling the negative cycle detection function
        # #if val==1 negative cycle exists, if val==0 negative cycle does not exist

        #Finding minimum element in our Green Route Matrix:
        min = 0
        for i in range(len(wmat)):
            x = wmat[i]
            for j in range(len(x)):
                if min > x[j]:
                    min = x[j]

        # If it contains negative cycle:
        if val > 0:
            jmat = []
            min = ((min) * (-1)) + 1
            for i in range(len(wmat)):
                x = wmat[i]
                for j in range(len(x)):
                    if x[j] != 0:
                        x[j] = x[j] + min
                jmat.append(x)
            wmat = jmat

            for i in range(len(wmat)):
                for j in range(len(wmat)):
                    if i != j and wmat[i][j] == INF:
                        wmat[i][j] = 0

    # Creating a numpy matrix for networkx as networkx takes numpy matrix only:
    B = np.matrix(wmat)  # converting wmat to numpy matrix
    plt.figure(figsize=(20, 12))
    G = nx.from_numpy_array(B)  # creating a graph with nodes and weighted edges using netwrokx by passing numpy matrix

    B = np.matrix(wmat)  # converting wmat to numpy matrix
    G = nx.from_numpy_array(B)  # creating a graph with nodes and weighted edges using netwrokx by passing numpy matrix

    # Finding Shortest path USING FLOYD WARSHALL THEORUM:
    path_lengths, _ = nx.floyd_warshall_predecessor_and_distance(G)
    path = nx.reconstruct_path(0, len(wmat)-1, path_lengths)

    if k ==0:
            print("For Green Route:\n", "Least Polluted path route(Green Route):", path)
            print("For Green Route:")
    if k == 1:
            print("For Yellow Route:\n", "Shortest Distance path route(Yellow):", path)
            print("For Yellow Route:")
    if k == 2:
            print("For Blue Route:\n", "Fastest Time path route(Blue) :", path)
            print("For Blue Route:")

    # FOR PRINTING ALL THE SHORTEST PATHS in GREEN ROUTE between all nodes:
    for i in range(len(wmat)):
        for j in range(len(wmat)):
            if i != j:
                locss = nx.reconstruct_path(i, j, path_lengths)
                print("(", i, " to ", j, ") Shortest Path: ---> ", locss, end = " ")
                for l in range(len(locss)):
                    if l == len(locss)-1:
                        print(loc[locss[l]])
                    else:
                        print(loc[locss[l]], end=" -> ")
    # -----------X----------#

    # Reverse Mapping the the Shortest path node back to our locations:
    print('Shortest Path from ', loc[path[0]], " to ", loc[path[len(path) - 1]], " is through ", end="")
    for i in range(1, len(path) - 1):
        print(" --> ", loc[path[i]], end=" ")

        if i == (len(path) - 2):
            print('\n')

    # Plotting edges:
    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 0]

    # Plotting the edges that are included in shortest path in green color:
    eshort1 = []
    eshort2 = []
    for i in range(len(elarge)):
        for j in range(len(path) - 1):
            if elarge[i] == (path[j], path[j + 1]):
                eshort1.append(elarge[i])
                eshortest = [(u, v) for (u, v, d) in G.edges(data=True) if eshort1[0] == (u, v)]
                eshort2.append(eshortest[0])
                eshort1 = []
                eshortest = []

    # Positions for all nodes - seed for reproducibility as used in networkx:
    pos = nx.spring_layout(G, seed=7)

    # Drawing nodes on the Graph:
    nx.draw_networkx_nodes(G, pos, node_size=700)

    # Drawing Edges on the graph:
    nx.draw_networkx_edges(G, pos, edgelist=elarge, width=6)
    if k==0:
     nx.draw_networkx_edges(G, pos, edgelist=eshort2, edge_color='green', width=6)  # This will plot shortest path edges
    elif k==1:
     nx.draw_networkx_edges(G, pos, edgelist=eshort2, edge_color='orange', width=6)  # This will plot shortest path edges
    elif k==2:
     nx.draw_networkx_edges(G, pos, edgelist=eshort2, edge_color='blue', width=6)  # This will plot shortest path edges

    # Drawing Node labels:
    node_label = {}
    list1 = []
    for i in range(len(loc)):
        list1 = loc[i]
        node_label[i] = []
        node_label[i].append(list1[0])
    nx.draw_networkx_labels(G, pos, node_label, font_size=20, font_family="sans-serif")

    # Drawing Edge weight labels
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels)

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    if k == 0:
        plt.savefig("GreenRouteGraph.png")
    if k == 1:
        plt.savefig("YellowRouteGraph.png")
    if k == 2:
        plt.savefig("BlueRouteGraph.png")
    # plt.show() to show graph in python

    # Selecting coordinates for shortest path found from matrix:
    print("Corrdinates of the Shortest Route: ")
    finalcoordx = []  # CONTAINS LONGITUDES
    finalcoordy = []  # CONTIANS LATITUDES

    for i in range(len(path)):
        finalcoordx.append(long[path[i]])
        finalcoordy.append(lat[path[i]])

    print(finalcoordx)
    print(finalcoordy)
    print("\n")
    # SHORTEST ROUTE ACCORD TO MATRIX:
    route_len = 0
    time_taken = 0
    shortest_routegx = []
    shortest_routeg=[]
    count = 0
    for i in range(len(path) - 1):
        # find the nearest node to the start location
        orig_node = ox.nearest_nodes(graph, finalcoordx[i], finalcoordy[i])
        # find the nearest node to the end location
        dest_node = ox.nearest_nodes(graph, finalcoordx[i + 1], finalcoordy[i + 1])
        
        # Compute shortest path for all cases
        shortest_routeg = nx.shortest_path(graph, orig_node, dest_node, weight=optimizer)

        # Get coordinates of the route nodes
        route_coords = [(graph.nodes[n]['y'], graph.nodes[n]['x']) for n in shortest_routeg]
        # (Route) --> For routing selected path on map:
        if k == 0 and count == 0:
            #shortest_route_map = ox.plot_route_folium(graph, shortest_routeg, color='green')
            #Green_route_map = ox.plot_route_folium(graph, shortest_routeg, color='green')
            
            # Collect coordinates of nodes in the route
            # Create Folium map centered at the start node
            shortest_route_map = folium.Map(location=route_coords[0], zoom_start=14)
            # Add the route as a green PolyLine
            folium.PolyLine(route_coords, color='green', weight=5, opacity=0.8).add_to(shortest_route_map)
            # Green_route_map as new map
            Green_route_map = folium.Map(location=route_coords[0], zoom_start=14)
            folium.PolyLine(route_coords, color='green', weight=5, opacity=0.8).add_to(Green_route_map)
            # Append nodes
            greennodes.extend(shortest_routeg)

        elif k == 0 and count>0:
            """
            shortest_routeg = nx.shortest_path(graph, orig_node, dest_node, weight=optimizer)
            shortest_route_map = ox.plot_route_folium(graph, shortest_routeg, route_map=shortest_route_map, color='green')
            Green_route_map = ox.plot_route_folium(graph, shortest_routeg,route_map=Green_route_map, color='green')
            for n in range(len(shortest_routeg)):
                greennodes.append(shortest_routeg[n])
            """
            folium.PolyLine(route_coords, color='green', weight=5, opacity=0.8).add_to(shortest_route_map)
            folium.PolyLine(route_coords, color='green', weight=5, opacity=0.8).add_to(Green_route_map)
    
            greennodes.extend(shortest_routeg)

        elif k == 1 and count == 0:
            """
            shortest_routeg = nx.shortest_path(graph, orig_node, dest_node, weight=optimizer)
            shortest_route_map = ox.plot_route_folium(graph, shortest_routeg,route_map=shortest_route_map, color='orange')
            Yellow_route_map = ox.plot_route_folium(graph, shortest_routeg, color='orange')
            for n in range(len(shortest_routeg)):
                yellownodes.append(shortest_routeg[n])
            """
            # Create or add to shortest_route_map with orange
            folium.PolyLine(route_coords, color='orange', weight=5, opacity=0.8).add_to(shortest_route_map)
    
            # Yellow_route_map as new map
            Yellow_route_map = folium.Map(location=route_coords[0], zoom_start=14)
            folium.PolyLine(route_coords, color='orange', weight=5, opacity=0.8).add_to(Yellow_route_map)
    
            yellownodes.extend(shortest_routeg)

        elif k == 1 and count>0:
            """
            shortest_routeg = nx.shortest_path(graph, orig_node, dest_node, weight=optimizer)
            shortest_route_map = ox.plot_route_folium(graph, shortest_routeg, route_map=shortest_route_map, color='orange')
            Yellow_route_map = ox.plot_route_folium(graph, shortest_routeg,route_map=Yellow_route_map, color='orange')
            for n in range(len(shortest_routeg)):
                yellownodes.append(shortest_routeg[n])
            """
            folium.PolyLine(route_coords, color='orange', weight=5, opacity=0.8).add_to(shortest_route_map)
            folium.PolyLine(route_coords, color='orange', weight=5, opacity=0.8).add_to(Yellow_route_map)
    
            yellownodes.extend(shortest_routeg)

        elif k == 2 and count == 0:
            """
            shortest_routeg = nx.shortest_path(graph, orig_node, dest_node, weight=optimizer)
            shortest_route_map = ox.plot_route_folium(graph, shortest_routeg, route_map=shortest_route_map, color='blue')
            Blue_route_map = ox.plot_route_folium(graph, shortest_routeg, color='blue')
            for n in range(len(shortest_routeg)):
                bluenodes.append(shortest_routeg[n])
            """
            folium.PolyLine(route_coords, color='blue', weight=5, opacity=0.8).add_to(shortest_route_map)
    
            Blue_route_map = folium.Map(location=route_coords[0], zoom_start=14)
            folium.PolyLine(route_coords, color='blue', weight=5, opacity=0.8).add_to(Blue_route_map)
    
            bluenodes.extend(shortest_routeg)

        else:
            """
            shortest_routeg = nx.shortest_path(graph, orig_node, dest_node, weight=optimizer)
            shortest_route_map = ox.plot_route_folium(graph, shortest_routeg, route_map=shortest_route_map,color='blue')
            Blue_route_map = ox.plot_route_folium(graph, shortest_routeg, route_map=Blue_route_map, color='blue')
            for n in range(len(shortest_routeg)):
                bluenodes.append(shortest_routeg[n])
            """
            folium.PolyLine(route_coords, color='blue', weight=5, opacity=0.8).add_to(shortest_route_map)
            folium.PolyLine(route_coords, color='blue', weight=5, opacity=0.8).add_to(Blue_route_map)
     
            bluenodes.extend(shortest_routeg)

        # For the Distance and Travel Time of Green Route:
        for u, v in zip(shortest_routeg[:-1], shortest_routeg[1:]):
            lengthgreen = round(speed.edges[(u, v, 0)]['length'])
            travel_timegreen = round(speed.edges[(u, v, 0)]['travel_time'])
            route_len = route_len + lengthgreen
            time_taken = time_taken + travel_timegreen

        # Again Adding Folium Markers For Route Locations:
        locshort = loc[path[i]]

        start_latlng = (finalcoordy[i], finalcoordx[i])
        start_marker = folium.Marker(
                location=start_latlng,
                popup=locshort[0],
                icon=folium.Icon(color='blue'))

        # Adding the circle marker to the map and Individual Maps:
        start_marker.add_to(shortest_route_map)

        name_marker = folium.map.Marker(
            [round(finalcoordy[i], 4), round(finalcoordx[i], 4)],
            icon=DivIcon(
                icon_size=(400, 400),
                icon_anchor=(0, 0),
                html='<div style="font-size: 10pt; color: black;">{}</div>'.format(locshort[0]),
            )
        )
        name_marker.add_to(shortest_route_map)
        if k ==0:
            locshort = loc[path[i]]

            start_latlng = (finalcoordy[i], finalcoordx[i])
            gstart_marker = folium.Marker(
                location=start_latlng,
                popup=locshort[0],
                icon=folium.Icon(color='blue'))

            # Adding the circle marker to the map and Individual Maps:
            gstart_marker.add_to(Green_route_map)

            gname_marker = folium.map.Marker(
                [round(finalcoordy[i], 4), round(finalcoordx[i], 4)],
                icon=DivIcon(
                    icon_size=(400, 400),
                    icon_anchor=(0, 0),
                    html='<div style="font-size: 10pt; color: black;">{}</div>'.format(locshort[0]),
                )
            )
            gname_marker.add_to(Green_route_map)
        elif k ==1:
            locshort = loc[path[i]]

            start_latlng = (finalcoordy[i], finalcoordx[i])
            ystart_marker = folium.Marker(
                location=start_latlng,
                popup=locshort[0],
                icon=folium.Icon(color='blue'))

            # Adding the circle marker to the map and Individual Maps:
            ystart_marker.add_to(Yellow_route_map)

            yname_marker = folium.map.Marker(
                [round(finalcoordy[i], 4), round(finalcoordx[i], 4)],
                icon=DivIcon(
                    icon_size=(400, 400),
                    icon_anchor=(0, 0),
                    html='<div style="font-size: 10pt; color: black;">{}</div>'.format(locshort[0]),
                )
            )
            yname_marker.add_to(Yellow_route_map)
        else:
            locshort = loc[path[i]]

            start_latlng = (finalcoordy[i], finalcoordx[i])
            ystart_marker = folium.Marker(
                location=start_latlng,
                popup=locshort[0],
                icon=folium.Icon(color='blue'))

            # Adding the circle marker to the map and Individual Maps:
            ystart_marker.add_to(Blue_route_map)

            yname_marker = folium.map.Marker(
                [round(finalcoordy[i], 4), round(finalcoordx[i], 4)],
                icon=DivIcon(
                    icon_size=(400, 400),
                    icon_anchor=(0, 0),
                    html='<div style="font-size: 10pt; color: black;">{}</div>'.format(locshort[0]),
                )
            )
            yname_marker.add_to(Blue_route_map)
        count = count + 1

    #Folium marker for destination:
    locshort = loc[path[len(path)-1]]
    start_latlng = (finalcoordy[len(finalcoordy) - 1], finalcoordx[len(finalcoordx) - 1])
    start_marker = folium.Marker(
            location=start_latlng,
            popup=locshort[0],
            icon=folium.Icon(color='blue'))

    # Adding the circle marker to the map:
    start_marker.add_to(shortest_route_map)

    dest_marker = folium.map.Marker(
            [round(finalcoordy[len(finalcoordy) - 1], 4), round(finalcoordx[len(finalcoordx) - 1], 4)],
            icon=DivIcon(
                icon_size=(400, 400),
                icon_anchor=(0, 0),
                html='<div style="font-size: 10pt; color: black;">{}</div>'.format(locshort[0]),
            )
    )
    dest_marker.add_to(shortest_route_map)
    # Optimizing Route Value:
    route_len = route_len / 1000
    route_len = round(route_len, 2)
    str1 = str(route_len)
    time_taken = int(time_taken / 60)
    str2 = str(time_taken)

    '''
    # Finding route length accoring to matrix:
    route_len = 0
    for i in range(len(path) - 1):
        route_len = route_len + lmat[path[i]][path[i + 1]]
    str1 = str(route_len)

    
    #Finding time taken accoring to matrix:
    time_taken=0
    for i in range(len(path)-1):
      time_taken = time_taken + tmat[path[i]][path[i+1]]
    str2 = str(time_taken)
    '''

    # For getting distances in start loc of map:
    t1, t2 = round(finalcoordy[0], 4), round(finalcoordx[0], 4)
    if k==1:
        msg1 = ""
        msg1 = msg1 + "Shortest Route:  " + str1 + " km, Time: " + str2 + " mins"
        t1 = t1 - 0.0035
        t2 = t2+0.0000
        message_marker = folium.map.Marker(
            [t1, t2],
            icon=DivIcon(
                icon_size=(400, 400),
                icon_anchor=(0, 0),
                html='<br> <div style="font-size: 10pt; color: orange;">{}</div><br>'.format(msg1),
            )
        )
        message_marker.add_to(shortest_route_map)

        ymessage_marker = folium.map.Marker(
            [t1, t2],
            icon=DivIcon(
                icon_size=(400, 400),
                icon_anchor=(0, 0),
                html='<br> <div style="font-size: 10pt; color: orange;">{}</div><br>'.format(msg1),
            )
        )
        ymessage_marker.add_to(Yellow_route_map)


        locshort = loc[path[len(path) - 1]]
        start_latlng = (finalcoordy[len(finalcoordy) - 1], finalcoordx[len(finalcoordx) - 1])
        ystart_marker = folium.Marker(
            location=start_latlng,
            popup=locshort[0],
            icon=folium.Icon(color='blue'))

        # Adding the circle marker to the map:
        ystart_marker.add_to(Yellow_route_map)

        ydest_marker = folium.map.Marker(
            [round(finalcoordy[len(finalcoordy) - 1], 4), round(finalcoordx[len(finalcoordx) - 1], 4)],
            icon=DivIcon(
                icon_size=(400, 400),
                icon_anchor=(0, 0),
                html='<div style="font-size: 10pt; color: black;">{}</div>'.format(locshort[0]),
            )
        )
        ydest_marker.add_to(Yellow_route_map)
        distances.append(route_len)
        time_travels.append(time_taken)

    elif k == 2:
        msg1 = ""
        msg1 = msg1 + "Fastest Route:   " + str1 + " km, Time: " + str2 + " mins"
        t1 = t1 - 0.0065
        t2 = t2 + 0.0000
        message_marker = folium.map.Marker(
            [t1, t2],
            icon=DivIcon(
                icon_size=(400, 400),
                icon_anchor=(0, 0),
                html='<br><div style="font-size: 10pt; color: blue;">{}</div><br>'.format(msg1),
            )
        )
        message_marker.add_to(shortest_route_map)

        bmessage_marker = folium.map.Marker(
            [t1, t2],
            icon=DivIcon(
                icon_size=(400, 400),
                icon_anchor=(0, 0),
                html='<br> <div style="font-size: 10pt; color: blue;">{}</div><br>'.format(msg1),
            )
        )
        bmessage_marker.add_to(Blue_route_map)

        locshort = loc[path[len(path) - 1]]
        start_latlng = (finalcoordy[len(finalcoordy) - 1], finalcoordx[len(finalcoordx) - 1])
        bstart_marker = folium.Marker(
            location=start_latlng,
            popup=locshort[0],
            icon=folium.Icon(color='blue'))

        # Adding the circle marker to the map:
        bstart_marker.add_to(Blue_route_map)

        bdest_marker = folium.map.Marker(
            [round(finalcoordy[len(finalcoordy) - 1], 4), round(finalcoordx[len(finalcoordx) - 1], 4)],
            icon=DivIcon(
                icon_size=(400, 400),
                icon_anchor=(0, 0),
                html='<div style="font-size: 10pt; color: black;">{}</div>'.format(locshort[0]),
            )
        )
        bdest_marker.add_to(Blue_route_map)

        distances.append(route_len)
        time_travels.append(time_taken)

    else:
        msg1 = ""
        msg1 = msg1 + "Greenest Route: " + str1 + " km, Time: " + str2 + " mins"
        t1 = t1 + 0.0002
        t2 = t2 + 0.0002
        message_marker = folium.map.Marker(
            [t1, t2],
            icon=DivIcon(
                icon_size=(400, 400),
                icon_anchor=(0, 0),
                html='<br> <div style="font-size: 10pt; color: green;">{}</div><br>'.format(msg1),
            )
        )
        message_marker.add_to(shortest_route_map)

        gmessage_marker = folium.map.Marker(
            [t1, t2],
            icon=DivIcon(
                icon_size=(400, 400),
                icon_anchor=(0, 0),
                html='<br> <div style="font-size: 10pt; color: green;">{}</div><br>'.format(msg1),
            )
        )
        gmessage_marker.add_to(Green_route_map)

        locshort = loc[path[len(path) - 1]]
        start_latlng = (finalcoordy[len(finalcoordy) - 1], finalcoordx[len(finalcoordx) - 1])
        gstart_marker = folium.Marker(
            location=start_latlng,
            popup=locshort[0],
            icon=folium.Icon(color='blue'))

        # Adding the circle marker to the map:
        gstart_marker.add_to(Green_route_map)

        gdest_marker = folium.map.Marker(
            [round(finalcoordy[len(finalcoordy) - 1], 4), round(finalcoordx[len(finalcoordx) - 1], 4)],
            icon=DivIcon(
                icon_size=(400, 400),
                icon_anchor=(0, 0),
                html='<div style="font-size: 10pt; color: black;">{}</div>'.format(locshort[0]),
            )
        )
        gdest_marker.add_to(Green_route_map)

        distances.append(route_len)
        time_travels.append(time_taken)


#PLOTTING THE MARKERS FOR COMMON ROUTES TO DISTINGUISH DIFFERENT ROUTES:
#Getting List of Nodes in Common Route of Green and Yellow Route:
gny = list(set(greennodes).intersection(yellownodes))

#Getting List of Nodes in Common Route of Green and Blue Route:
gnb = list(set(greennodes).intersection(bluenodes))

#Getting List of Nodes in Common Route of Yellow and Blue Route:
ynb = list(set(yellownodes).intersection(bluenodes))

#Getting List of Nodes in Common Route of Green and Yellow and Blue Route:
gnynb = list(set(gny).intersection(bluenodes))

"""
print(greennodes)
print(yellownodes)
print(bluenodes)
print(gnb)
print(gny)
print(ynb)
print(gnynb)
"""

#Plotting Markers for Green Color which are on same route as with Yellow Route:
for i in range(len(gny)):
    if i %2==0:
        longitudef = graph.nodes[gny[i]]['x']  # lon
        latitudef = graph.nodes[gny[i]]['y']  # lat

        folium.CircleMarker(location=[latitudef, longitudef],
                            radius=1,
                            weight=4,
                            color="green"
                            ).add_to(shortest_route_map)

#Plotting Markers for Green Color which are on same route as with Blue Route:
for i in range(len(gnb)):
    longitudef = graph.nodes[gnb[i]]['x'] # lon
    latitudef = graph.nodes[gnb[i]]['y']  # lat

    folium.CircleMarker(location=[latitudef, longitudef],
                        radius=1,
                        weight=4,
                        color = "green"
                        ).add_to(shortest_route_map)

#Plotting Markers for Yellow Color which are on same route as with Blue Route:
for i in range(len(ynb)):
    longitudef = graph.nodes[ynb[i]]['x'] # lon
    latitudef = graph.nodes[ynb[i]]['y']  # lat

    folium.CircleMarker(location=[latitudef, longitudef],
                        radius=1,
                        weight=4,
                        color = "orange"
                        ).add_to(shortest_route_map)

#Plotting Markers for Green Color and Yellow Color where all 3 Routes are common:
colorcount =0
for i in range(len(gnynb)):
    longitudef = graph.nodes[gnynb[i]]['x'] # lon
    latitudef = graph.nodes[gnynb[i]]['y']  # lat
    if i % 2 == 0 and colorcount == 0:
        color = "green"
        colorcount = 1
    elif i % 2 == 0 and colorcount == 1:
        color ="orange"
        colorcount = 2
    elif i % 2 == 0 and colorcount == 2:
        color ="blue"
        colorcount = 0

    folium.CircleMarker(location=[latitudef, longitudef],
                        radius=1,
                        weight=4,
                        color = color
                        ).add_to(shortest_route_map)


#PRINTING FINAL DISTANCE AND TIME FOR ALL 3 ROUTES
print("\nRoutes distances and Time Travels:")
for i in range(len(distances)):
    if i ==0:#Green Route
        print("Green Route Distance: ", distances[i], " km")
        print("Time for Green route:  ", time_travels[i], " mins","\n")
    if i ==1: #Yellow Route
        print("Yellow (Shortest) Route Distance: ", distances[i], " km")
        print("Time for Yellow(Shortest) route: ", time_travels[i], " mins","\n")
    if i ==2: #Blue Route
        print("Blue(Fastest) Route Distance: ", distances[i], " km")
        print("Time for Blue(Fastest) route:  ", time_travels[i], " mins","\n")

#SAVING ALL 4 MAPS IN HTML MAP:
shortest_route_map.save('AllRoutesmap.html')
Green_route_map.save('GreenRouteMap.html')
Yellow_route_map.save('YellowRouteMap.html')
Blue_route_map.save('BlueRouteMap.html')


'''
gmat = [[0, -2, -1, -1, 0, 0, 0],
         [-2, 0, 1, 1, 0, 0, 0],
         [-1, 1, 0, 0, -1, 0, 0],
         [-1, 1, 0, 0, -1, 0, 0],
         [0, 0, -1, -1, 0, 0, -1], 
         [0, 0, 0, 0, 0, 0, -1],
         [0, 0, 0, 0, -1, -1, 0]]
'''


