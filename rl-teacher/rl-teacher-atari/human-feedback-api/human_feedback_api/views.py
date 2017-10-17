import random
from collections import namedtuple
from datetime import timedelta, datetime

from django import template
from django.db.models import Q
from django.shortcuts import get_object_or_404
from django.shortcuts import render
from django.utils import timezone

from human_feedback_api.models import Comparison
from human_feedback_api.models import SortTree
from human_feedback_api.models import Clip

import human_feedback_api.redblack_tree as redblack

register = template.Library()

ExperimentResource = namedtuple("ExperimentResource", ['name', 'num_responses', 'started_at', 'pretty_time_elapsed'])

def _pretty_time_elapsed(start, end):
    total_seconds = (end - start).total_seconds()
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return ("{:0>2}:{:0>2}:{:0>2}".format(int(hours), int(minutes), int(seconds)))

def _build_experiment_resource(experiment_name):
    comparisons = Comparison.objects.filter(experiment_name=experiment_name, responded_at__isnull=False)
    try:
        started_at = comparisons.order_by('-created_at').first().created_at
        pretty_time_elapsed = _pretty_time_elapsed(started_at, timezone.now())
    except AttributeError:
        started_at = None
        pretty_time_elapsed = None
    return ExperimentResource(
        name=experiment_name,
        num_responses=comparisons.count(),
        started_at=started_at,
        pretty_time_elapsed=pretty_time_elapsed
    )

def _all_comparisons(experiment_name, use_locking=False):
    not_responded = Q(responded_at__isnull=True)

    if use_locking:
        cutoff_time = timezone.now() - timedelta(minutes=2)
        not_in_progress = Q(shown_to_tasker_at__isnull=True) | Q(shown_to_tasker_at__lte=cutoff_time)
        finished_uploading_media = Q(created_at__lte=datetime.now() - timedelta(seconds=2))  # Give time for upload
        ready = not_responded & not_in_progress & finished_uploading_media
    else:
        ready = not_responded

    # Sort by priority, then put newest labels first
    return Comparison.objects.filter(ready, experiment_name=experiment_name).order_by('-priority', '-created_at')

def index(request):
    binary_tree_experiments = set([name for name in SortTree.objects.filter(parent=None).order_by().values_list('experiment_name', flat=True)])
    all_comparison_experiments = [name for name in Comparison.objects.order_by().values_list('experiment_name', flat=True)]
    other_experiments = set(all_comparison_experiments) - binary_tree_experiments
    return render(request, 'index.html', context={
        'binary_tree_experiments': binary_tree_experiments, 'other_experiments': other_experiments})

def list_comparisons(request, experiment_name):
    comparisons = Comparison.objects.filter(experiment_name=experiment_name).order_by('responded_at', '-priority')
    return render(request, 'list.html', context=dict(comparisons=comparisons, experiment_name=experiment_name))

def display_comparison(comparison):
    """Mark comparison as having been displayed"""
    comparison.shown_to_tasker_at = timezone.now()
    comparison.save()

def ajax_response(request, experiment_name):
    """Update a comparison with a response"""

    POST = request.POST
    comparison_id = POST.get("comparison_id")
    debug = True

    comparison = Comparison.objects.get(pk=comparison_id)

    # Update the values
    comparison.response = POST.get("response")
    comparison.responded_at = timezone.now()

    if debug:
        print("Answered comparison {} with {}".format(comparison_id, comparison.response))

    comparison.full_clean()  # Validation
    comparison.save()

    # If this comparison belongs to a sorting tree, run the tree logic...
    _sorting_logic(experiment_name)

    comparisons = list(_all_comparisons(experiment_name)[:1])
    for comparison in comparisons:
        display_comparison(comparison)
    if debug:
        print("{}".format([x.id for x in comparisons]))
        if comparison:
            print("Requested {}".format(comparison.id))
    return render(request, 'ajax_response.html', context={
        'comparisons': comparisons,
        'experiment': _build_experiment_resource(experiment_name)
    })

def show_comparison(request, comparison_id):
    comparison = get_object_or_404(Comparison, pk=comparison_id)
    return render(request, 'show_feedback.html', context={"feedback": comparison})

def respond(request, experiment_name):
    # This only does things if the experiment uses a sorting tree.
    _sorting_logic(experiment_name)

    # The response interface queues up some comparisons and handles the logic of serving up
    # new ones to that queue via AJAX, so the user doesn't have to refresh or wait between
    # labeling comparisons.

    number_of_queued_comparisons = 3
    comparisons = list(_all_comparisons(experiment_name)[:number_of_queued_comparisons])
    for comparison in comparisons:
        display_comparison(comparison)

    return render(request, 'responses.html', context={
        'comparisons': comparisons,
        'experiment': _build_experiment_resource(experiment_name)
    })

def all_clips(request, environment_id):
    return render(request, 'all_clips.html', context={"clips": Clip.objects.filter(environment_id=environment_id)})

# Sorting tree logic:
def _handle_comparison_on_node(comp, node, experiment_name):
    print("Handling", comp, "for", node)
    # First mark the comparison as no longer relevant
    comp.relevant_to_pending_clip = False
    comp.save()
    # Get the clip being compared
    clip = comp.left_clip
    print("Working with", clip)
    # Mark the clip as no longer pending for this node
    node.pending_clips.remove(clip)
    # Move the clip to the right place
    if comp.response in ["left", "right"]:
        try:
            print("Trying to move", clip, "down the tree!")
            redblack.move_clip_down(node, clip, comp.response)
        except redblack.NewNodeNeeded as need_new_node:
            print("We need a new node for", clip)
            # The tree may have shifted. First verify that the clip has been compared to all parents.
            check_node = node.parent
            while check_node:
                if not Comparison.objects.filter(tree_node=check_node, left_clip=clip):
                    need_new_node = False
                    check_node.pending_clips.add(clip)
                    print("Oh! Just kidding! The upstream parent,", check_node, "doesn't have a comparison for", clip)
                    print("Reassinging the clip to the upstream parent.")
                    break
                check_node = check_node.parent
            if need_new_node:
                new_node = SortTree(
                    experiment_name=node.experiment_name,
                    is_red=True,
                    parent=node,
                )
                new_node.save()
                new_node.bound_clips.add(clip)
                print("Created", new_node)
                print("New Node", new_node, "is being seeded with", clip)
                if need_new_node.on_the_left:
                    node.left = new_node
                else:
                    node.right = new_node
                node.save()
                redblack.rebalance_tree(new_node)
    else:  # Assume tie
        node.bound_clips.add(clip)
        print(clip, 'being assigned to', node)

def _handle_node_with_pending_clips(node, experiment_name):
    comparisons_to_handle = Comparison.objects.filter(tree_node=node, relevant_to_pending_clip=True).exclude(response=None)
    if comparisons_to_handle:
        print(node, "has comparisons to handle!")
        _handle_comparison_on_node(comparisons_to_handle[0], node, experiment_name)
        return True
    elif not Comparison.objects.filter(tree_node=node, relevant_to_pending_clip=True):
        print(node, "needs a new comparison!")
        # Make a comparison, since there are no relevant ones for this node.
        clip1 = node.pending_clips.all()[0]
        clip2 = random.choice(node.bound_clips.all())
        print("Let's make a comparison between", clip1, "and", clip2)
        comparison = Comparison(
            experiment_name=experiment_name,
            left_clip=clip1,
            right_clip=clip2,
            response_kind='left_or_right',
            priority=0.1 if node.parent is None else 1.0,  # De-prioritize comparisons on the root
            tree_node=node,
            relevant_to_pending_clip=True,
        )
        print(comparison, "created!")
        comparison.full_clean()
        comparison.save()
    # else:
    #   We're waiting for the user to label the comparison for this node
    return False

def _sorting_logic(experiment_name):
    print("Sorting logic start for ", experiment_name)
    run_logic = True
    while run_logic:
        print("Logic loop")
        run_logic = False
        # Look to generate comparisons from the tree
        active_tree_nodes = SortTree.objects.filter(experiment_name=experiment_name).exclude(pending_clips=None)
        for node in active_tree_nodes:
            print("Logic for", node)
            tree_changed = _handle_node_with_pending_clips(node, experiment_name)
            if tree_changed:
                print("Tree changed!")
                # If the tree changed we want to immediately stop the logic and restart to avoid conncurrent writes
                run_logic = True
                break

# Tree visualization code:
def _get_visnodes(node, depth, tree_position, what_kind_of_child_i_am):
    max_depth = depth
    results = [{
        'id': 'visnode%s' % node.id,
        'name': node.id,
        'bound_clips': [clip.media_url for clip in node.bound_clips.all()],
        'tree_position': tree_position,  # If the root pos=1, this ranges (0, 2)
        'depth': depth,
        'color': '#A00' if node.is_red else 'black',
        'text_color': 'white',
        'what_kind_of_child_i_am': what_kind_of_child_i_am,
        'num_bound_clips': len(node.bound_clips.all()),
        'num_pending_clips': len(node.pending_clips.all()),
    }]
    if node.right:
        right_position = tree_position + (0.5**(depth + 1))
        sub_visnodes, max_subdepth = _get_visnodes(node.right, depth + 1, right_position, 'right')
        results += sub_visnodes
        max_depth = max(max_depth, max_subdepth)
    if node.left:
        left_position = tree_position - (0.5**(depth + 1))
        sub_visnodes, max_subdepth = _get_visnodes(node.left, depth + 1, left_position, 'left')
        results += sub_visnodes
        max_depth = max(max_depth, max_subdepth)
    return results, max_depth

def _set_visnode_position_data(visnodes, max_depth, clip_width):
    clip_plus_frame_width = clip_width + 20
    clip_plus_frame_height = clip_width + 116
    largest_row = 2 ** max_depth
    total_width = clip_plus_frame_width * largest_row
    total_height = (max_depth + 1) * clip_plus_frame_height
    for vn in visnodes:
        vn['top_edge'] = vn['depth'] * clip_plus_frame_height
        vn['left_edge'] = total_width * (vn['tree_position'] / 2)
        vn['x_position'] = vn['left_edge'] + (clip_plus_frame_width / 2)
        vn['y_position'] = vn['top_edge'] + (clip_plus_frame_height / 2)
        shift = 0.5 ** vn['depth']  # How much we have to move over in tree_position to get to the parent
        if vn['what_kind_of_child_i_am'] == "left":
            vn['parent_x_pos'] = vn['x_position'] + (total_width * (shift / 2))
            vn['parent_y_pos'] = vn['y_position'] - clip_plus_frame_height
        elif vn['what_kind_of_child_i_am'] == "right":
            vn['parent_x_pos'] = vn['x_position'] - (total_width * (shift / 2))
            vn['parent_y_pos'] = vn['y_position'] - clip_plus_frame_height
        else:
            vn['parent_x_pos'] = vn['x_position']
            vn['parent_y_pos'] = vn['y_position']
    return total_width, total_height

def tree(request, experiment_name):
    root = SortTree.objects.get(experiment_name=experiment_name, parent=None)
    visnodes, max_depth = _get_visnodes(root, depth=0, tree_position=1, what_kind_of_child_i_am=None)
    dim = _set_visnode_position_data(visnodes, max_depth, 84)
    return render(request, 'tree.html', context={"tree": visnodes, "total": {"width": dim[0], "height": dim[1]}})
