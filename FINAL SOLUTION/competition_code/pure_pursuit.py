import numpy as np

def normalize_rad(rad: float) -> float:
    return (rad + np.pi) % (2 * np.pi) - np.pi

class PurePursuit:
    def __init__(self, path):
        self.path = path

    def sgn(self, x):
        return -1 if x < 0 else 1
    
    def step(self, currentPos, currentHeading, lookAheadDis, LFindex, vehicle_velocity_norm):
        currentX, currentY = currentPos
        lastFoundIndex = LFindex
        
        for i in range(LFindex, len(self.path) + LFindex):
            currentPoint = self.path[i % len(self.path)]
            nextPoint = self.path[(i + 1) % len(self.path)]
            
            x1, y1 = currentPoint[0] - currentX, currentPoint[1] - currentY
            x2, y2 = nextPoint[0] - currentX, nextPoint[1] - currentY
            dx, dy = x2 - x1, y2 - y1
            dr = np.hypot(dx, dy)
            D = x1 * y2 - x2 * y1
            discriminant = (lookAheadDis ** 2) * (dr ** 2) - D ** 2
            
            if discriminant >= 0:
                sqrt_disc = np.sqrt(discriminant)
                sol_pts = [
                    [((D * dy + self.sgn(dy) * dx * sqrt_disc) / dr ** 2) + currentX,
                     ((-D * dx + abs(dy) * sqrt_disc) / dr ** 2) + currentY],
                    [((D * dy - self.sgn(dy) * dx * sqrt_disc) / dr ** 2) + currentX,
                     ((-D * dx - abs(dy) * sqrt_disc) / dr ** 2) + currentY]
                ]
                
                minX, minY = min(currentPoint[0], nextPoint[0]), min(currentPoint[1], nextPoint[1])
                maxX, maxY = max(currentPoint[0], nextPoint[0]), max(currentPoint[1], nextPoint[1])

                in_range_pts = [pt for pt in sol_pts if minX <= pt[0] <= maxX and minY <= pt[1] <= maxY]
                
                if in_range_pts:
                    if len(in_range_pts) == 2:
                        dists = [np.hypot(pt[0] - nextPoint[0], pt[1] - nextPoint[1]) for pt in in_range_pts]
                        goalPt = in_range_pts[np.argmin(dists)]
                    else:
                        goalPt = in_range_pts[0]

                    if np.hypot(goalPt[0] - nextPoint[0], goalPt[1] - nextPoint[1]) < np.hypot(currentX - nextPoint[0], currentY - nextPoint[1]):
                        lastFoundIndex = i % len(self.path)
                        break
                    else:
                        lastFoundIndex = (i + 1) % len(self.path)
                else:
                    goalPt = self.path[lastFoundIndex]
            else:
                goalPt = self.path[lastFoundIndex]

        target_heading = np.arctan2(goalPt[1] - currentY, goalPt[0] - currentX)
        turnError = normalize_rad(target_heading - currentHeading)
        turnVel = -18.0 / np.sqrt(vehicle_velocity_norm) * turnError / np.pi

        return goalPt, lastFoundIndex, turnVel
