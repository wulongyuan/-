#include "BodyType.h"

bool skeleton::operator == (const skeleton &i)
{
	return (bodyPoint[BodyData_head] == i.bodyPoint[BodyData_head] && 
		bodyPoint[BodyData_leftHand] == i.bodyPoint[BodyData_leftHand] &&
		bodyPoint[BodyData_rightHand] == i.bodyPoint[BodyData_rightHand] &&
		bodyPoint[BodyData_chest] == i.bodyPoint[BodyData_chest] &&
		bodyPoint[BodyData_hip] == i.bodyPoint[BodyData_hip] &&
		bodyPoint[BodyData_leftFoot] == i.bodyPoint[BodyData_leftFoot] &&
		bodyPoint[BodyData_rightFoot] == i.bodyPoint[BodyData_rightFoot]);
}


bool PersonData::operator==(const PersonData &i)
{
	return (index == i.index && skeletonData == i.skeletonData);
}
