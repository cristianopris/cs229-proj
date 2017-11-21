using UnityEngine;
using System.Collections;

public class HoopBehaviour : MonoBehaviour
{
	public Collision lastCollision; 


	// Use this for initialization
	public void restart()
	{
		lastCollision = null;
		gameObject.GetComponent<Renderer>().material.color = Color.white;
	}
	
	// Update is called once per frame
	void Update ()
	{
	
	}

	void OnCollisionEnter(Collision collisionInfo)
	{
		lastCollision = collisionInfo;
		gameObject.GetComponent<Renderer>().material.color = Color.red;
	}
		
}

