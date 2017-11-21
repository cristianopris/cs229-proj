using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Ball3DAgent : Agent
{
    [Header("Specific to Ball3D")]
    public GameObject ball;
	public GameObject hoop;

	[HideInInspector]
	HoopBehaviour hoopBeh;

	bool gotReward = false;

	public override void InitializeAgent()
	{
		Debug.Log ("Cameras : " + this.observations);
		hoopBeh = hoop.GetComponent<HoopBehaviour> ();
		//resetOnDone = false;
	}

	Collision lastCollision; 

    public override List<float> CollectState()
    {
        List<float> state = new List<float>();

		//Platform rotation
        state.Add(gameObject.transform.rotation.z);
        state.Add(gameObject.transform.rotation.x);
        
		//Ball position
		state.Add((ball.transform.position.x - gameObject.transform.position.x) / 5f);
        state.Add((ball.transform.position.y - gameObject.transform.position.y) / 5f);
        state.Add((ball.transform.position.z - gameObject.transform.position.z) / 5f);
        
		//Ball velocity
		state.Add(ball.transform.GetComponent<Rigidbody>().velocity.x / 5f);
        state.Add(ball.transform.GetComponent<Rigidbody>().velocity.y / 5f);
        state.Add(ball.transform.GetComponent<Rigidbody>().velocity.z / 5f);
        return state;
    }


	void OnCollisionEnter(Collision collisionInfo)
	{
//		print("Detected collision between " + gameObject.name + " and " + collisionInfo.collider.name);
//		print("There are " + collisionInfo.contacts.Length + " point(s) of contacts");
		print("Collision velocity: " + collisionInfo.relativeVelocity + " magnitude: " + collisionInfo.relativeVelocity.magnitude);
		lastCollision = collisionInfo;


		// how much the character should be knocked back
		var magnitude =  collisionInfo.relativeVelocity.magnitude * - 2f;

		// calculate force vector
		var force = transform.position - collisionInfo.transform.position;
		// normalize force vector to get direction only and trim magnitude
		force.Normalize();

		//Debug.Log ("AddForce: force: " + force + " magnitude: " + magnitude);

		collisionInfo.rigidbody.AddForce(force * magnitude);	

		gotReward = false;
	}

    // to be implemented by the developer
    public override void AgentStep(float[] act)
	{
		reward = 0;
		if (brain.brainParameters.actionSpaceType == StateType.continuous) {

			float clip = 15f; //2f;

			float action_z = act [0];
			if (action_z > clip) {
				action_z = clip;
			}
			if (action_z < -clip) {
				action_z = -clip;
			}

			if ((gameObject.transform.rotation.z < 0.25f && action_z > 0f) ||
			             (gameObject.transform.rotation.z > -0.25f && action_z < 0f)) {
				gameObject.transform.Rotate (new Vector3 (0, 0, 1), action_z);
			}

			float action_x = act [1];
			if (action_x > clip) {
				action_x = clip;
			}
			if (action_x < -clip) {
				action_x = -clip;
			}

			if ((gameObject.transform.rotation.x < 0.25f && action_x > 0f) ||
			             (gameObject.transform.rotation.x > -0.25f && action_x < 0f)) {
				gameObject.transform.Rotate (new Vector3 (1, 0, 0), action_x);
			}


			if (done == false) {
				if (hoopBeh.lastCollision != null) {

					reward = 0.1f;
					Debug.Log ("Reward: " + reward);
					hoopBeh.lastCollision = null;
					gotReward = true;
				}
			}
		}
			

		if ((ball.transform.position.y - gameObject.transform.position.y) < -5f 
//			|| Mathf.Abs (ball.transform.position.x - gameObject.transform.position.x) > 3f ||
//				Mathf.Abs (ball.transform.position.z - gameObject.transform.position.z) > 3f
		) {
			if (gotReward) {
				gotReward = false;
				restart();
			} else {
				gotReward = false;
				done = true;
				reward = -1f;
			}
//			done=true;
//			reward = -1f;
		}
    }



	public override void AgentReset() {
		restart();
	}

    void restart()
    {
		//randomize platform rotation
        gameObject.transform.rotation = new Quaternion(0f, 0f, 0f, 0f);
        gameObject.transform.Rotate(new Vector3(1, 0, 0), Random.Range(-10f, 10f));
        gameObject.transform.Rotate(new Vector3(0, 0, 1), Random.Range(-10f, 10f));

		//randomize ball position
		//if (id == 0) 
		{
			ball.GetComponent<Rigidbody> ().velocity = new Vector3 (0f, 0f, 0f);
			//ball.transform.position = new Vector3(Random.Range(-1.5f, 1.5f), 4f, Random.Range(-1.5f, 1.5f)) + gameObject.transform.position;
			ball.transform.position = new Vector3 (Random.Range (-2f, 2f), 4f, Random.Range (-2f, 2f)) + gameObject.transform.position;
			//ball.transform.position = new Vector3 (0f, 4f, 0f) + gameObject.transform.position;
		}
		hoopBeh.restart();
	}

	float sigmoid(float x, float c, float a) {
		return 1f/(1f + Mathf.Exp(-a*(x-c)));
	}
}
