import random
import math
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def initial_route(n):
    return {'Mon': {'08:00': (n, n)},
            'Tue': {'08:00': (n, n)},
            'Wed': {'08:00': (n, n)},
            'Thur': {'08:00': (n, n)},
            'Fri': {'08:00': (n, n)}}


def create_schedule():
    """Creating schedule divided by time slot of 15 minutes, from 8:00 on Monday to 18:00 on Friday."""
    # Suppose working from 8 am to 18 pm, from Monday to Friday, each time slot spends 15 minutes.
    working_days = ['Mon', 'Tue', 'Wed', 'Thur', 'Fri']
    start_time = datetime.strptime("08:00", "%H:%M")
    end_time = datetime.strptime("18:00", "%H:%M")
    time_slot = timedelta(minutes=15)

    # Create dictionary of schedule divided by time slots.
    schedule = {}
    for day in working_days:
        day_schedule = []
        current_time = start_time
        while current_time <= end_time:
            day_schedule.append(current_time.strftime("%H:%M"))
            current_time += time_slot
        schedule[day] = day_schedule
    return schedule


def generate_random_points(n, region):
    """Generate location that completely random."""
    x = np.random.randint(0, region - 1, n)
    y = np.random.randint(0, region - 1, n)
    loc_lst = list(zip(x, y))
    return loc_lst


def generate_uniform_points(n, region):
    """Generate location with uniform distribution."""
    x = np.random.uniform(0, region - 1, n)
    y = np.random.uniform(0, region - 1, n)
    x = np.floor(x).astype(int)
    y = np.floor(y).astype(int)
    loc_lst = list(zip(x, y))
    return loc_lst


def generate_cluster_points(n, region, clusters, spread):
    """Generate location with cluster distribution."""
    points_per_cluster = n // clusters
    x = []
    y = []

    for _ in range(clusters):
        cluster_x = np.random.uniform(0, region - 1)
        cluster_y = np.random.uniform(0, region - 1)
        x_cluster = np.random.normal(cluster_x, spread, points_per_cluster)
        y_cluster = np.random.normal(cluster_y, spread, points_per_cluster)
        x.extend(x_cluster)
        y.extend(y_cluster)

    remaining_points = n - len(x)
    if remaining_points > 0:
        x_rem = np.random.normal(cluster_x, spread, remaining_points)
        y_rem = np.random.normal(cluster_y, spread, remaining_points)
        x.extend(x_rem)
        y.extend(y_rem)

    x = np.floor(x).astype(int)
    y = np.floor(y).astype(int)
    points_list = list(zip(x, y))
    return points_list


def generate_combined_points(n, region, ratio):
    """Generate location with combined distribution."""
    cluster_num = int(n * ratio)
    uniform_num = n - cluster_num

    uniform_loc_list = generate_uniform_points(uniform_num, region)
    cluster_loc_list = generate_cluster_points(cluster_num, region, 2, ratio)
    loc_list = uniform_loc_list + cluster_loc_list

    return loc_list


def generate_patient(schedule, loc):
    """Create information of patient, including location and available time slot."""
    working_days = ['Mon', 'Tue', 'Wed', 'Thur', 'Fri']

    # Randomly choosing coordinate of location.
    location = (int(loc[0]), int(loc[1]))

    # Generating 5 available time slots.
    slots_num = 5
    available_time_slots = []
    for _ in range(slots_num):
        day = random.choice(working_days)
        time_slot = random.choice(schedule[day])
        available_time_slots.append((day, time_slot))

    # Create dictionary to store location and available time slots of patient.
    patient = {
        "location": location,
        "available_time_slots": available_time_slots
    }
    return patient


def travel_time(location1, location2):
    """calculate travel time that each unit spends 3 minutes."""
    return math.ceil(math.dist(location1, location2) * 3)


def available(schedule, day, patient_slot, treatment_time, nurse_route, patient_location):
    """Check if patient_slot is available. If so, return both nearest appointments that before and after this slot."""

    # To ensure that treatment ends before the end of work time.
    if schedule[day].index(patient_slot) + treatment_time - 1 >= len(schedule[day]):
        return None, None, None, None

    # divided current appointments by inserted slot time.
    less_slots = []
    greater_slots = []
    for time in nurse_route[day].keys():
        if time <= patient_slot:
            less_slots.append(time)
        else:
            greater_slots.append(time)
    # find nearest less time slot.
    if less_slots:
        nearest_less = max(less_slots)
    else:
        nearest_less = None
    # find nearest greater time slot.
    if greater_slots:
        nearest_greater = min(greater_slots)
    else:
        nearest_greater = None

    # The nearest_less is guaranteed since nurse location is initialized at (0, 0).
    loc_less = nurse_route[day][nearest_less]
    travel_time_from_previous = travel_time(loc_less, patient_location)
    # Only one less_slot means nurse start at 8:00, otherwise it should finishes from the end point of last treatment.
    if len(less_slots) == 1:
        last_datetime = datetime.strptime(nearest_less, "%H:%M")
    else:
        last_datetime = datetime.strptime(nearest_less, "%H:%M") + timedelta(minutes=15 * treatment_time)
    # The travel time from the end of last time point should not exceed the starting time of this treatment.
    if datetime.strptime(patient_slot, "%H:%M") < timedelta(minutes=travel_time_from_previous) + last_datetime:
        return None, None, None, None

    # The nearest_greater equals to current location and time if insertion is at the tail of schedule.
    if nearest_greater:
        loc_greater = nurse_route[day][nearest_greater]
        travel_time_to_next = travel_time(patient_location, loc_greater)
        current_datetime = datetime.strptime(patient_slot, "%H:%M") + timedelta(minutes=15 * treatment_time)
        # The travel time from the end of this time point should not exceed the starting time of next treatment.
        if datetime.strptime(nearest_greater, "%H:%M") < current_datetime + timedelta(minutes=travel_time_to_next):
            return None, None, None, None
    else:
        loc_greater = patient_location

    return nearest_less, loc_less, nearest_greater, loc_greater


def BC(starting, ending, treatment_time, travel=1):
    """Calculates best case capacity that describes how many visits can be inserted, suppose travel spends one slot."""
    last_end = datetime.strptime(starting, "%H:%M")
    next_begin = datetime.strptime(ending, "%H:%M") - timedelta(minutes=15 * travel)
    best_capacity1 = (next_begin - last_end) / timedelta(minutes=15 * treatment_time) - 1
    best_capacity2 = (next_begin - datetime.strptime(starting, "%H:%M")) / (
        timedelta(minutes=15 * (treatment_time + 1))) - 1
    return min(int(best_capacity1), int(best_capacity2))


def reduced_visit_capacity(nearest_less, nearest_greater, patient_slot, treatment_time):
    """For one inserted appointment, the slot is divided into two parts.
       The sum of two parts with the inserted appointment can be less than original capacity."""
    old_vc = BC(nearest_less, nearest_greater, treatment_time)
    new_vc = BC(nearest_less, patient_slot, treatment_time) + BC(patient_slot, nearest_greater, treatment_time)
    return old_vc - new_vc - 1


def flexible_capacity(starting, ending, treatment_time):
    """Flexible capacity shows the extra slots that can be used to make room for more flexible arrangement."""
    last_end = datetime.strptime(starting, "%H:%M")
    slots_num = (datetime.strptime(ending, "%H:%M") - last_end) / timedelta(minutes=15)
    BC_val = BC(starting, ending, treatment_time)
    FC_val = slots_num - (treatment_time + 1) * (BC_val + 1)
    return int(FC_val)


def reduced_flexible_capacity(starting, ending, slot, treatment_time):
    """Flexible capacity shows the extra slots that can be used to make room for more flexible arrangement."""
    Re_FC = 0
    BC_val = BC(starting, ending, treatment_time)
    FC_val = flexible_capacity(starting, ending, treatment_time)
    if BC_val >= 1:
        BC1 = BC(starting, slot, treatment_time)
        BC2 = BC(slot, ending, treatment_time)
        slot_time = datetime.strptime(slot, "%H:%M")
        slots_num1 = (slot_time - datetime.strptime(starting, "%H:%M")) / timedelta(minutes=15) - treatment_time
        slots_num2 = (datetime.strptime(ending, "%H:%M") - slot_time) / timedelta(minutes=15) - treatment_time
        if BC1 == 0 & int(slots_num1) > 1:
            Re_FC = min(Re_FC + slots_num1, FC_val)
        if BC2 == 0 & int(slots_num2) > 1:
            Re_FC = min(Re_FC + slots_num2, FC_val)
    return int(Re_FC)


def daily_workload_capacity(day, nurse_route):
    """Daily workload capacity is the amount of appointments in one day."""
    return len(nurse_route[day])


def update_schedule_distance(schedule, patient, nurse_route_distance, route_dis_week, accept):
    """For each new appointment, this function implements distance heuristic for optimal arrangement."""
    treatment = 2  # 30 minutes (2 time slots) for one treatment.

    # Distance heuristic calculates extra distance spent on new location.
    costs = {}
    for day, slot in patient['available_time_slots']:  # Calculate cost if available.
        _, place1, _, place2 = available(schedule, day, slot, treatment, nurse_route_distance,
                                         patient['location'])
        if place1 is not None:
            new1 = travel_time(place1, patient['location'])
            new2 = travel_time(patient['location'], place2)
            old = travel_time(place1, place2)
            cost = new1 + new2 - old
            costs[(day, slot)] = (cost, new1, new2)
        else:
            continue

    # If available choice exists, then choose the option with minimal cost.
    if len(costs) != 0:
        min_cost = min(cost for cost, new1, new2 in costs.values())
        min_slots = [(option, info) for option, info in costs.items() if info[0] == min_cost]
    else:
        min_slots = []

    # If more than one minimal cost, then choose the one that nearest to other appointment.
    if len(min_slots) >= 2:
        best_day, best_slot = min_slots[0][0]
        min_val = min_slots[0][1][1]
        for i in range(len(min_slots)):
            # new1, new2 stands for distance between current location and nearest location.
            cost, new1, new2 = min_slots[i][1]
            temp = min(new1, new2)
            if temp < min_val:
                min_val = temp
                best_day, best_slot = min_slots[i][0]
    elif len(min_slots) == 1:
        best_day, best_slot = min_slots[0][0]
    else:
        best_day, best_slot = None, None

    if best_day and best_slot:
        nurse_route_distance[best_day][best_slot] = patient['location']
        route_dis_week[best_day][best_slot] = patient['location']
        accept += 1

    return nurse_route_distance, route_dis_week, accept


def update_schedule_capacity(schedule, patient, nurse_route_capacity, route_capa_week, accept):
    """For each new appointment, this function implements capacity heuristic for optimal arrangement."""
    treatment = 2  # 30 minutes (2 time slots) for one treatment.

    # Capacity heuristic calculates different type of reduced capacity.
    costs = {}

    for day, slot in patient['available_time_slots']:  # Calculate cost if available.
        time1, _, time2, _ = available(schedule, day, slot, treatment, nurse_route_capacity,
                                       patient['location'])
        if time1 is not None:
            time2 = schedule[day][len(schedule[day]) - 1] if time2 is None else time2
            rvc = reduced_visit_capacity(time1, time2, slot, treatment)
            rfc = reduced_flexible_capacity(time1, time2, slot, treatment)
            dwc = daily_workload_capacity(day, nurse_route_capacity)
            costs[(day, slot)] = (rvc, rfc, dwc)
        else:
            continue

    # If available choice exists, then choose the option with minimal cost, priority order: rvc, rfc, dwc.
    if len(costs) != 0:
        min_rvc = min(rvc for rvc, _, _ in costs.values())
        min_slots = [(option, info) for option, info in costs.items() if info[0] == min_rvc]
        if len(min_slots) >= 2:
            min_rfc = min(rfc for _, rfc, _ in costs.values())
            min_slots = [(option, info) for option, info in costs.items() if info[1] == min_rfc]
            if len(min_slots) >= 2:
                min_dwc = min(dwc for _, _, dwc in costs.values())
                min_slots = [(option, info) for option, info in costs.items() if info[2] == min_dwc]
    else:
        min_slots = []

    best_day, best_slot = (min_slots[0][0][0], min_slots[0][0][1]) if len(min_slots) >= 1 else (None, None)

    if best_day and best_slot:
        nurse_route_capacity[best_day][best_slot] = patient['location']
        route_capa_week[best_day][best_slot] = patient['location']
        accept += 1

    return nurse_route_capacity, route_capa_week, accept


def merge_dict(routes_x):
    """Merge 3 dictionaries into the total dictionary that removed completed appointments."""
    new_total = routes_x[0].copy()
    for day in new_total.keys():
        new_total[day].update(routes_x[1][day])
        new_total[day].update(routes_x[2][day])
    return new_total.copy()


def sort_route(nurse_route):
    """Sort route in ascending order of time, for better readability."""
    for day, info in nurse_route.items():
        nurse_route[day] = {time: loc for time, loc in sorted(nurse_route[day].items())}
    return nurse_route


def calcu_distance(route):
    """calculate travelling distance """
    total_dis = 0
    for day in route.keys():
        locs = route[day]
        sorted_times = sorted(locs.keys())
        for i in range(1, len(sorted_times)):
            pre = locs[sorted_times[i - 1]]
            cur = locs[sorted_times[i]]
            total_dis += math.dist(pre, cur)
        # Add distance traveling back from the last location to home.
        final = locs[sorted_times[-1]]
        home = (0, 0)
        total_dis += math.dist(final, home)
    return total_dis


def operation_per_week(schedule, routes_d, routes_c, total_route_d, total_route_c, num, n, nurse_loc, loc_list):
    """For each week, remove appointments that has ended and then add new appointment."""
    # Suppose each appointment lasts for 3 weeks, then re-initialize schedule 3 weeks before.
    th = num % 3
    if num >= 3:
        routes_d[th] = initial_route(nurse_loc)
        routes_c[th] = initial_route(nurse_loc)
        total_route_d = merge_dict(routes_d)
        total_route_c = merge_dict(routes_c)

    acd, acc = 0, 0
    # suppose we have n patients. each time implement both heuristic separately.
    for i in range(n):
        loc = num * n + i
        patient = generate_patient(schedule, loc_list[loc])
        total_route_d, routes_d[th], acd = update_schedule_distance(schedule, patient, total_route_d, routes_d[th], acd)
        total_route_c, routes_c[th], acc = update_schedule_capacity(schedule, patient, total_route_c, routes_c[th], acc)

    # Sort route in ascending order of time, for better readability.
    total_route_d = sort_route(total_route_d)
    total_route_c = sort_route(total_route_c)

    # Calculate final travelling distance of nurse.
    total_dis_d = calcu_distance(total_route_d)
    total_dis_c = calcu_distance(total_route_c)

    print(f"WEEK {num + 1}:")
    print(f"for distance heuristic:")
    print(f"Nurse Route on Monday: {total_route_d['Mon']}")
    print(f"Nurse Route on Tuesday: {total_route_d['Tue']}")
    print(f"Nurse Route on Wednesday: {total_route_d['Wed']}")
    print(f"Nurse Route on Thursday: {total_route_d['Thur']}")
    print(f"Nurse Route on Friday: {total_route_d['Fri']}\n")

    print(f"for capacity heuristic:")
    print(f"Nurse Route on Monday: {total_route_c['Mon']}")
    print(f"Nurse Route on Tuesday: {total_route_c['Tue']}")
    print(f"Nurse Route on Wednesday: {total_route_c['Wed']}")
    print(f"Nurse Route on Thursday: {total_route_c['Thur']}")
    print(f"Nurse Route on Friday: {total_route_c['Fri']}\n")

    return total_dis_d / 5, acd / n, total_dis_c / 5, acc / n


def operation(region, patient_num, distribution):
    """Start whole process and set different parameters."""
    schedule = create_schedule()
    nurse_loc = region // 2  # nurse's location at the center of region.
    total_route_d = initial_route(nurse_loc)
    total_route_c = initial_route(nurse_loc)
    routes_d = {0: initial_route(nurse_loc), 1: initial_route(nurse_loc), 2: initial_route(nurse_loc)}
    routes_c = {0: initial_route(nurse_loc), 1: initial_route(nurse_loc), 2: initial_route(nurse_loc)}
    t_d, d_a, t_c, c_a = 0, 0, 0, 0

    if distribution == "U":
        loc_list = generate_uniform_points(patient_num * 10, region)
    elif distribution == "C":
        loc_list = generate_cluster_points(patient_num * 10, region, clusters=2, spread=0.5)
    elif distribution == "UC":
        loc_list = generate_combined_points(patient_num * 10, region, ratio=0.5)
    else:
        loc_list = generate_random_points(patient_num * 10, region)

    for n in range(10):
        td, da, tc, ca = operation_per_week(schedule, routes_d, routes_c, total_route_d, total_route_c, n,
                                            patient_num, nurse_loc, loc_list)
        t_d += td
        d_a += da
        t_c += tc
        c_a += ca

    print(f"Summary:")
    print(f"On average, travelling distance under distance heuristic is {t_d / 10} per day;")
    print(f"And its acceptance rate is {d_a / 10}.\n")
    print(f"On average, travelling distance under capacity heuristic is {t_c / 10} per day;")
    print(f"And its acceptance rate is {c_a / 10}.\n")
    return t_d / 10, d_a / 10, t_c / 10, c_a / 10


#operation(region=10, patient_num=8, distribution="U")

# test case
numbers = []
td, da, tc, ca = 0, 0, 0, 0
for _ in range(50):
    t_d, d_a, t_c, c_a = operation(region=20, patient_num=25, distribution="U")
    td += t_d
    da += d_a
    tc += t_c
    ca += c_a
    numbers.append((c_a - d_a)/d_a)
print(f"Summary:")
print(f"On average, travelling distance under distance heuristic is {td / 50} per day;")
print(f"And its acceptance rate is {da / 50}.\n")
print(f"On average, travelling distance under capacity heuristic is {tc / 50} per day;")
print(f"And its acceptance rate is {ca / 50}.\n")
plt.plot(numbers)
plt.title('Change of Acceptance Rate from DH to CH')
plt.xlabel('Week')
plt.ylabel('Rate')
plt.show()

