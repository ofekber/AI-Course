import search
import random
import math
import itertools
import json

ids = ["111111111", "111111111"]
#state , רשימה של מיקומי אוכלי מוות-0. מקומות וחיים של קוסמים-1. מקומות של הורוקרוסים 2 מספר הורקוקסים 3 הרגנו אותו

class HarryPotterProblem(search.Problem):
    count=0
    map=[]
    horcruxe_board=[]
    num_of_horcruxes_left=0
    num_of_horcruxes=0
    wizards=[]
    death_eaters=[]
    horcruxes=[]
    the_one_that_shound_not_be_named_locetion=(0,0)
    the_one_that_shound_not_be_named_dead=0
    death_eaters_path=[]
    top_board=0
    left_board=0
    distenes_board=[]
    grops=[]
    time_to_finsh=[]
    """This class implements a medical problem according to problem description file"""
    def __init__(self, initial):
        self.map=initial["map"]
        self.top_board=len(self.map[0])
        self.left_board=len(self.map)
        self.wizards=initial["wizards"]
        self.death_eaters=initial["death_eaters"]
        self.horcruxes=initial["horcruxes"]
        self.horcruxe_board,self.num_of_horcruxes_left=self.horcruxes_loc()
        self.num_of_horcruxes=self.num_of_horcruxes_left
        for i, row in enumerate(self.map):
            for j, value in enumerate(row):
                if value == 'V':

                    self.the_one_that_shound_not_be_named_locetion=(i,j)

        self.death_eaters_path = {death_eater: 0 for death_eater in self.death_eaters}
        arr=list(range(0,self.num_of_horcruxes_left))
        state = {
            "death_eaters_path": self.death_eaters_path,
            "wizards": self.wizards,
            "horcruxe_board": self.horcruxe_board,
            "num_of_horcruxes_left": self.num_of_horcruxes_left,
            "numbers_left":arr,
        }
        self.state = json.dumps(state, indent=3)
        search.Problem.__init__(self, self.state)

        self.distenes_board=self.my_bfs(self.num_of_horcruxes_left,initial["horcruxes"])
        self.horcruxes.append(self.the_one_that_shound_not_be_named_locetion)
        self.grops=self.beast_group(self.wizards,self.horcruxes)
        self.group_dis(self.horcruxes)
     #   print(self.distenes_board)
     #   print(self.grops)
     #   print(self.time_to_finsh)
    def option_cost(self,option,wizerds,horcruxe):
        max=0
        time=0
        i=0
        wizerds_arr=list(wizerds.values())
        wizerds_key=list(wizerds.keys())
        for group in option:
            time=0
            x,y=wizerds_arr[i][0]
            for target in group:
                time=time+self.distenes_board[x][y][target]+1
                x,y=horcruxe[target]
            if(wizerds_key[i]=="Harry Potter"):
                time=time+self.distenes_board[x][y][self.num_of_horcruxes]+1
            else:
                time=time+2
            if time>max:
                max=time
            i=i+1
        return max

    def beast_group(self,wizerds,horcruxes):
        wizerds_key=list(wizerds.keys())
        options=self.brotForce(self.num_of_horcruxes,len(self.wizards))
        min=float('inf')
        bestop=options[0]
        curent=None
        for option in options:
            curent=self.option_cost(option,wizerds,horcruxes)
            if curent<min:
                min=curent
                bestop=option
        #add voll to harry
        i=0
        for key in wizerds_key:
            if key=="Harry Potter":
                bestop[i].append(self.num_of_horcruxes)
            i=i+1
        return bestop
    def group_dis(self,horcruxes):
        self.time_to_finsh=[0]*(self.num_of_horcruxes+1)
        for group in self.grops:
            revers_group=group[::-1]
            distens=1
            i=0
            prives=-1
            for hor in revers_group:
               if(i==0):
                    self.time_to_finsh[hor]=1
                    distens=1
                    i=i+1
               else:
                   x,y=horcruxes[hor]
                   distens=distens+self.distenes_board[x][y][prives]+1
                   self.time_to_finsh[hor]=distens
               prives=hor
    def h(self, node):
        help=json.loads(node.state)
        self.death_eaters_path=help["death_eaters_path"]
        self.death_eaters_new_location()
        cost=0
        death_eaters_board=self.death_eaters_new_postion()
        horcruxe_board=help["horcruxe_board"]
        wizerds=help["wizards"]
        num_of_horcruxes_left=help["num_of_horcruxes_left"]
        horcruxes_left=help["numbers_left"]
        if(num_of_horcruxes_left==-11):
            return float('inf')
        if(num_of_horcruxes_left==0):
            x,y=wizerds["Harry Potter"][0]
            life=wizerds["Harry Potter"][1]
            return self.distenes_board[x][y][self.num_of_horcruxes]
        if(num_of_horcruxes_left==-10):
            return 0
        return self.optimal_cost(wizerds,horcruxes_left)
    def targets(self,horcruxes_left):
        targets=[-1]*len(self.wizards)
        i=0
        for group in self.grops:
            for x in group:
                if(x in horcruxes_left):
                    targets[i]=x
                    break
            i=i+1
        return targets
    def optimal_cost(self,wizerds,horcruxes_left):
        targets=self.targets(horcruxes_left)
        max=0
        current=0
        wizerds_arr=list(wizerds.values())
        for i in range(len(targets)):
            if(targets[i]!=-1):
                x,y=self.horcruxes[targets[i]]
                wiz_x,wiz_y=wizerds_arr[i][0]

                current=self.time_to_finsh[targets[i]]+self.distenes_board[wiz_x][wiz_y][targets[i]]
            if(current>max):
                max=current
        return max
    def more_wiz_cost(self,wizards,horcruxes_left):
        for hor in horcruxes_left:
            closest_wiz=None
            min=float('inf')
            for wiz in wizards:
                x,y=wizards["wiz"][0]
                distens=self.distenes_board[x][y][hor]
                if(distens<min):
                    closest_wiz=wiz
                    min=distens
            

    def more_hor_cost(self):
        return 0
    def same_amount(self):
        return 0
    def my_bfs(self,num_of_hourcocses,hourocses):
        board=self.map
        optiones=[]
        # Define the reference array (arr2)
        # Create an array (arr) of the same size as arr2, filled with empty arrays

        arr = [[[float('inf')] * (num_of_hourcocses+1) for _ in row] for row in board]
        num_of_horcruxes=0
        hourocses.append([self.the_one_that_shound_not_be_named_locetion[0],self.the_one_that_shound_not_be_named_locetion[1]])
        for horx in hourocses:
            start_x,start_y=horx
            optiones.append([start_x,start_y,0])
            while len(optiones)>0:
                curent=optiones[0]
                x=curent[0]
                y=curent[1]
                optiones.remove(curent)
                if( arr[x][y][num_of_horcruxes]==float('inf')):

                    arr[x][y][num_of_horcruxes]=curent[2]
                    if(x<self.left_board-1):
                        if(self.map[x+1][y]=='P'):
                            optiones.append([x+1,y,curent[2]+1])
                    if(x>0):
                        if(self.map[x-1][y]=='P'):
                            optiones.append([x-1,y,curent[2]+1])

                    if(y<self.top_board-1):
                        if(self.map[x][y+1]=='P'):
                            optiones.append([x,y+1,curent[2]+1])
                    if(y>0):
                        if(self.map[x][y-1]=='P' ):
                            optiones.append([x,y-1,curent[2]+1])
            num_of_horcruxes=num_of_horcruxes+1
        hourocses.remove([self.the_one_that_shound_not_be_named_locetion[0],self.the_one_that_shound_not_be_named_locetion[1]])
        return arr


    def actions(self, state):

        help=json.loads(state)
        self.death_eaters_path=help["death_eaters_path"]
        death_eaters_board=self.death_eaters_new_postion()
        horcruxe_board=help["horcruxe_board"]
        wizards=help["wizards"]
       # print("wizerd board",wizards)
       # print("death_eaters_path",death_eaters_board)
        num_of_horcruxes_left=help["num_of_horcruxes_left"]
        moves = [[] for _ in range(len(self.wizards))]
        i=0
        for wizard in self.wizards:

            x,y=wizards[wizard][0]
            life=wizards[wizard][1]
            #allow here poter move to vold
            if(wizard=="Harry Potter"):
                if(num_of_horcruxes_left==0):
                    if((life>1 or life==1 and death_eaters_board[x][y]==0)):
                        if((x,y)==self.the_one_that_shound_not_be_named_locetion):
                            moves[i].append(["kill","Harry Potter"])
                    if(x<self.left_board-1):
                        if(self.map[x+1][y]=='V' and (life>1 or life==1 and death_eaters_board[x+1][y]==0)):
                            moves[i].append(["move",wizard,(x+1,y)])
                    if(x>0):
                        if(self.map[x-1][y]=='V' and (life>1 or life==1 and death_eaters_board[x-1][y]==0)):
                            moves[i].append(["move",wizard,(x-1,y)])
                    if(y<self.top_board-1):
                        if(self.map[x][y+1]=='V' and (life>1 or life==1 and death_eaters_board[x][y+1]==0)):
                            moves[i].append(["move",wizard,(x,y+1)])
                    if(y>0):
                        if(self.map[x][y-1]=='V' and (life>1 or life==1 and death_eaters_board[x][y-1]==0)):
                            moves[i].append(["move",wizard,(x,y-1)])

            if(x<self.left_board-1):
                if(self.map[x+1][y]=='P' and (life>1 or life==1 and death_eaters_board[x+1][y]==0)):
                    moves[i].append(["move",wizard,(x+1,y)])
            if(x>0):
                if(self.map[x-1][y]=='P' and (life>1 or life==1 and death_eaters_board[x-1][y]==0)):
                    moves[i].append(["move",wizard,(x-1,y)])
            if(y<self.top_board-1):
                if(self.map[x][y+1]=='P' and (life>1 or life==1 and death_eaters_board[x][y+1]==0)):
                    moves[i].append(["move",wizard,(x,y+1)])
            if(y>0):
                if(self.map[x][y-1]=='P' and (life>1 or life==1 and death_eaters_board[x][y-1]==0)):
                    moves[i].append(["move",wizard,(x,y-1)])
            if((life>1 or life==1 and death_eaters_board[x][y]==0)):
                if(len(horcruxe_board[x][y])>0):
                    moves[i].append(["destroy",wizard,horcruxe_board[x][y][0]])
                else:
                    moves[i].append(["wait",wizard])
            if(len(moves[i])==0):
                moves[i].append(["wait",wizard])
            i=i+1

        #print("moves each arr i is the move wizeerd i can do ",moves)
        actiones = list(itertools.product(*moves))

        return actiones
        """Return the valid actions that can be executed in the given state."""
    def horcruxes_loc(self):
        board = [[[] for _ in row] for row in self.map]
        count=0
        i=0
        for horcruxe in self.horcruxes:
            x,y=horcruxe
            board[x][y].append(i)
            count=count+1
            i=i+1
        return board,count
    def death_eaters_new_location(self):
        for death_eater in self.death_eaters:
            if self.death_eaters_path[death_eater]>-1:
                self.death_eaters_path[death_eater]+=1
                if(self.death_eaters_path[death_eater]==len(self.death_eaters[death_eater])-1):
                    self.death_eaters_path[death_eater]=self.death_eaters_path[death_eater]*-1
                else:
                    self.death_eaters_path[death_eater]-=1
    def death_eaters_new_postion(self):
        board = [[0 for _ in row] for row in self.map]
        for death_eater in self.death_eaters:
            x,y=self.death_eaters[death_eater][self.death_eaters_path[death_eater]]
            board[x][y]=1
        return board
    def result(self, state, action):
        help=json.loads(state)
        #print("before")
        #print(help)
        #print(action)
        #create deep copy plzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz

        self.death_eaters_path=help["death_eaters_path"]
        self.death_eaters_new_location()
        death_eaters_board=self.death_eaters_new_postion()
        horcruxe_board=help["horcruxe_board"]
        wizerds=help["wizards"]
        horcruxes_left=help["numbers_left"]
        num_of_horcruxes_left=help["num_of_horcruxes_left"]
        for action in action:
            if(action[0]=="move"):
                x,y=action[2]
                wizerds[action[1]][0]=x,y
                if(death_eaters_board[x][y]==1):
                    wizerds[action[1]][1]=wizerds[action[1]][1]-1
                    if(wizerds[action[1]][1]==0):
                        num_of_horcruxes_left=-11
            if(action[0]=="wait"):
                x,y=wizerds[action[1]][0]
                if(death_eaters_board[x][y]==1):
                    wizerds[action[1]][1]=wizerds[action[1]][1]-1
                    if(wizerds[action[1]][1]==0):
                        num_of_horcruxes_left=-11
            if(action[0]=="destroy"):
                x,y=wizerds[action[1]][0]
                if action[2] in horcruxe_board[x][y]:
                    horcruxe_board[x][y].remove(action[2])
                    horcruxes_left.remove(action[2])
                    num_of_horcruxes_left=num_of_horcruxes_left-1
                if(death_eaters_board[x][y]==1):
                    wizerds[action[1]][1]=wizerds[action[1]][1]-1
                    if(wizerds[action[1]][1]==0):
                        num_of_horcruxes_left=-11
            if(action[0]=="kill"):
                x,y=wizerds[action[1]][0]
                if num_of_horcruxes_left!=-11:
                    num_of_horcruxes_left=-10
                if(death_eaters_board[x][y]==1):
                    wizerds[action[1]][1]=wizerds[action[1]][1]-1
                    if(wizerds[action[1]][1]==0):
                        num_of_horcruxes_left=-11
        """Return the state that results from executing the given action in the given state."""
        state = {
            "death_eaters_path": self.death_eaters_path,
            "wizards": wizerds,
            "horcruxe_board": horcruxe_board,
            "num_of_horcruxes_left": num_of_horcruxes_left,
            "numbers_left": horcruxes_left,
        }
      #  print("after")
       # print(state)
        self.state = json.dumps(state, indent=3)

        return self.state


    def goal_test(self, state):
        """Return True if the state is a goal state."""
        "not cheking if someone die"
        data_dict = json.loads(state)

        if "num_of_horcruxes_left" in state:
            if data_dict["num_of_horcruxes_left"]==-10:
                return True
        return False
    def cost(self, wizard,horcruxe_board):
        cost=0
        stuff_to_destroy=self.stuff_to_destroy(horcruxe_board)
        target=[]

        for stuff in stuff_to_destroy:
            if(len(wizard)>0):
                min=10000000000000000000000
                beast_wiz=None
                curent=0
                for wiz in wizard:
                    x,y=wizard[wiz][0]
                    life=wizard[wiz][1]

                    x_t,y_t=stuff
                    curent=abs(x-x_t)+abs(y-y_t)-life
                    if(curent<min):
                        min=curent
                        beast_wiz=wiz
                cost=cost+min
                del wizard[beast_wiz]
            else:
                cost=cost+self.left_board+self.top_board
        return cost
           # wizard
           #     v_x,v_y=self.the_one_that_shound_not_be_named_locetion

    def stuff_to_destroy(self, horcruxe_board):
        stuff_to_destroy=[]
        i=0
        j=0
        for x in horcruxe_board:
            for y in x:
                if(len(y)>0):
                    stuff_to_destroy.append((i,j))
                j=j+1
            i=i+1
        return stuff_to_destroy
    def h2(self, node):
        """
        Heuristic function for A* search.
        Estimates the minimum number of moves needed to reach the goal.
        """

        help=json.loads(node.state)
        self.death_eaters_path=help["death_eaters_path"]
        self.death_eaters_new_location()
        death_eaters_board=self.death_eaters_new_postion()
        horcruxe_board=help["horcruxe_board"]
        wizerds=help["wizards"]
        num_of_horcruxes_left=help["num_of_horcruxes_left"]
        if(num_of_horcruxes_left==-11):
            x=1000000000000000000000000000000
        if(num_of_horcruxes_left==0):
            x,y=wizerds["Harry Potter"][0]
            life=wizerds["Harry Potter"][1]
            v_x,v_y=self.the_one_that_shound_not_be_named_locetion
            return abs(x-v_x)+abs(y-v_y)+abs(life)
        if(num_of_horcruxes_left==-10):
            return 0
        return self.cost(wizerds,horcruxe_board)*num_of_horcruxes_left
       # if(len(wizerds>self.num_of_horcruxes_left)):
    def h1(self, node):
        return 0
        help=json.loads(node.state)
        self.death_eaters_path=help["death_eaters_path"]
        self.death_eaters_new_location()
        cost=0
        death_eaters_board=self.death_eaters_new_postion()
        horcruxe_board=help["horcruxe_board"]
        wizerds=help["wizards"]
        num_of_horcruxes_left=help["num_of_horcruxes_left"]
        if(num_of_horcruxes_left==-11):
            x=1000000000000000000000000000000
        if(num_of_horcruxes_left==0):
            x,y=wizerds["Harry Potter"][0]
            life=wizerds["Harry Potter"][1]
            v_x,v_y=self.the_one_that_shound_not_be_named_locetion
            return abs(x-v_x)+abs(y-v_y)+abs(life)
        if(num_of_horcruxes_left==-10):
            return 0
        if(num_of_horcruxes_left<len(wizerds)):
            x,y=wizerds["Harry Potter"][0]
            life=wizerds["Harry Potter"][1]
            v_x,v_y=self.the_one_that_shound_not_be_named_locetion
            cost=abs(x-v_x)+abs(y-v_y)+abs(life)
            del wizerds["Harry Potter"]
        return self.cost(wizerds,horcruxe_board)*num_of_horcruxes_left+cost
    def to_close(self,arr,num,x,y):
        x_bot=x-num
        x_top=x+num
        y_bot=y-num
        y_top=y+num
        while(x_bot<=x_top):
            while(y_bot<=y_top):
                arr.append([x_bot,y_bot])
                y_bot=y_bot+1
            x_bot=x_bot+1

    def cost2(self, wizard,horcruxe_board):
        cost=0
        stuff_to_destroy=self.stuff_to_destroy(horcruxe_board)
        in_arr=[]
        help=1
        target_arr=[]
        i=2

        for wiz in wizard:
                x,y=wizard[wiz][0]
                life=wizard[wiz][1]
                min=10000000000000000000000
                beast_stuff=None
                curent=0
                help=1
                while(len(stuff in stuff_to_destroy and stuff not in target_arr[i])==0):
                    i=i-1
                if(i==-1):
                    break
                for stuff in stuff_to_destroy and stuff not in target_arr[i]:
                    x_t,y_t=stuff
                    curent=abs(x-x_t)+abs(y-y_t)-life
                    if(curent<min):
                        min=curent
                        beast_stuff=stuff
                cost=cost+min
                self.to_close(in_arr[2],2,x_t,y_t)
                self.to_close(in_arr[1],1,x_t,y_t)
                self.to_close(in_arr[0],0,x_t,y_t)


        return cost


    def brotForce(self,x, y):
        """
        Divide numbers from 0 to x into y groups and return all possible groupings.

        :param x: The maximum number in the range (0 to x).
        :param y: The number of groups.
        :return: A list of all possible groupings.
        """
        numbers = list(range(x))

        # Generate all possible partitions where numbers are used exactly once
        def partitions(numbers, y):
            # Generate all assignments of numbers to groups
            for assignment in itertools.product(range(y), repeat=len(numbers)):
                groups = [[] for _ in range(y)]
                for number, group in zip(numbers, assignment):
                    groups[group].append(number)
                yield groups

        # Generate and return all possible groupings
        results = [grouping for grouping in partitions(numbers, y)]
        return results
def create_harrypotter_problem(game):
    return HarryPotterProblem(game)
